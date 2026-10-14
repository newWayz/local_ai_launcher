import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

/**
 * CPU Auxiliary Microservices Extension for Pi.
 *
 * Exposes the local zero-GPU-overhead CPU tools directly to Pi's tool registry:
 * 1. cpu_rerank: Cross-encoder semantic snippet ranking (bge-reranker-v2-m3 on Port 8082 / Gateway 8080, ~25ms)
 * 2. cpu_decide: Sub-15ms discrete choice evaluation (von System One on Port 8083 / Gateway 8080, ~12ms)
 * 3. cpu_tripwire: Sub-10ms safety & command blast-radius tripwire (von on Port 8083, ~8ms)
 * 4. cpu_multislot: Multi-slot state extraction & planning (Decider-4B v2.1 on Port 8084, ~50ms)
 */
export default function (pi: ExtensionAPI) {
  // 1. CPU Reranker Tool (~25ms cross-encoder)
  pi.registerTool({
    name: "cpu_rerank",
    label: "CPU Reranker",
    description:
      "Rerank a list of code snippets, passage hunks, or search results by semantic relevance against a query using the local CPU cross-encoder (bge-reranker-v2-m3). Extremely fast (~25ms on CPU). Use this when grep or file searches return multiple matches to identify the top passages without blowing up LLM context. Note: Whole source files (>3,000 tokens) automatically bypass reranking; pass focused snippets (<800-1,000 tokens).",
    promptSnippet: "cpu_rerank: Reranks candidate code or text passages against a query using local CPU cross-encoder (~25ms)",
    promptGuidelines: [
      "When a search or grep returns multiple matches, use cpu_rerank to score and pick the most relevant snippets. Do NOT pass whole files (>3,000 tokens); pass extracted candidate passages."
    ],
    parameters: Type.Object({
      query: Type.String({ description: "The specific intent, question, or symbol being searched for" }),
      documents: Type.Array(Type.String(), { description: "Array of candidate text or code snippets to evaluate (<800-1,000 tokens each; whole files >3,000 tokens bypass reranking)" }),
      top_n: Type.Optional(Type.Integer({ description: "Number of top results to return (default: 5)" }))
    }),
    async execute(toolCallId, params, signal) {
      const ports = [8082, 8080];
      let lastError: any = null;

      for (const port of ports) {
        try {
          const url = `http://127.0.0.1:${port}/v1/rerank`;
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query: params.query,
              documents: params.documents,
              top_n: params.top_n ?? 5
            }),
            signal
          });
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status}: ${await res.text()}`);
            continue;
          }
          const data = (await res.json()) as { results: any[]; model?: string };
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(data.results || data, null, 2)
              }
            ],
            details: { count: data.results?.length ?? 0, port }
          };
        } catch (err: any) {
          lastError = err;
        }
      }

      return {
        content: [
          {
            type: "text",
            text: `Error connecting to CPU Reranker on ports 8082/8080: ${lastError?.message || lastError}`
          }
        ],
        isError: true
      };
    }
  });

  // 2. CPU ONNX Decision Gater (~14ms discrete classifier)
  pi.registerTool({
    name: "cpu_decide",
    label: "CPU von Gater",
    description:
      "Perform sub-15ms discrete choice evaluation (PASS/FAIL, ALLOW/BLOCK, OPTION_A/OPTION_B) using the local CPU von gater (wfzyx/von on port 8083). Zero GPU VRAM overhead. Use for lint checks, safety guardrails, syntax checks, or boolean decision triage.",
    promptSnippet: "cpu_decide: Fast discrete evaluation (PASS/FAIL, ALLOW/BLOCK) via local CPU von Gater (~12ms)",
    promptGuidelines: [
      "Use cpu_decide for fast deterministic evaluations or gate decisions instead of performing multi-token reasoning turns."
    ],
    parameters: Type.Object({
      prompt: Type.String({ description: "Context, rule, or test output to evaluate" }),
      candidate_options: Type.Array(Type.String(), {
        description: "Allowed discrete choices (e.g. ['PASS', 'FAIL'], ['ALLOW', 'BLOCK'])"
      }),
      temperature: Type.Optional(Type.Number({ description: "Softmax sharpness (default 1.45)" }))
    }),
    async execute(toolCallId, params, signal) {
      const ports = [8083, 8080];
      let lastError: any = null;

      for (const port of ports) {
        try {
          const url = `http://127.0.0.1:${port}/v1/decide`;
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompt: params.prompt,
              candidate_options: params.candidate_options,
              temperature: params.temperature ?? 1.45
            }),
            signal
          });
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status}: ${await res.text()}`);
            continue;
          }
          const data = (await res.json()) as { choice: string; confidence: number };
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(data, null, 2)
              }
            ],
            details: { choice: data.choice, confidence: data.confidence, port }
          };
        } catch (err: any) {
          lastError = err;
        }
      }

      return {
        content: [
          {
            type: "text",
            text: `Error connecting to CPU ONNX Gater on ports 8083/8080: ${lastError?.message || lastError}`
          }
        ],
        isError: true
      };
    }
  });

  // 3. CPU CLM Tool Shortlister (~14ms dual-encoder with CSLS & negation masking)
  pi.registerTool({
    name: "cpu_shortlist",
    label: "CPU Tool Shortlister",
    description:
      "Shortlist candidate tools using the local CPU CLM dual-encoder (~14ms). Applies lexical negation masking ('avoid', 'never') and CSLS hubness mitigation to extract the top-3 to top-5 most relevant tools for a given state.",
    promptSnippet: "cpu_shortlist: Shortlist candidate tools for a state using local CPU CLM (~14ms)",
    promptGuidelines: [
      "Use cpu_shortlist when facing large tool sets (10+ candidates) to extract the most relevant tools for the current task."
    ],
    parameters: Type.Object({
      state: Type.String({ description: "Current conversation state or task intent" }),
      tools: Type.Array(
        Type.Object({
          name: Type.String({ description: "Tool name" }),
          description: Type.Optional(Type.String({ description: "Tool description" }))
        }),
        { description: "List of candidate tools to evaluate" }
      ),
      top_k: Type.Optional(Type.Integer({ description: "Shortlist target size (default: 3)" }))
    }),
    async execute(toolCallId, params, signal) {
      const ports = [8083, 8080];
      let lastError: any = null;

      for (const port of ports) {
        try {
          const url = `http://127.0.0.1:${port}/v1/shortlist`;
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              state: params.state,
              tools: params.tools,
              top_k: params.top_k ?? 3
            }),
            signal
          });
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status}: ${await res.text()}`);
            continue;
          }
          const data = await res.json();
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(data, null, 2)
              }
            ],
            details: {
              shortlisted: data.shortlisted_tools?.map((t: any) => t.name) ?? [],
              expansion_k: data.expansion_k,
              port
            }
          };
        } catch (err: any) {
          lastError = err;
        }
      }

      return {
        content: [
          {
            type: "text",
            text: `Error connecting to CPU Shortlister on ports 8083/8080: ${lastError?.message || lastError}`
          }
        ],
        isError: true
      };
    }
  });

  // 4. CPU Action Vector Registration & Cache Precomputation
  pi.registerTool({
    name: "cpu_register_tools",
    label: "CPU Pre-cache Tools",
    description:
      "Precomputes and caches action embeddings for candidate tools in DDR5 RAM (~400 KB) using the CPU CLM engine. Eliminates per-turn tool re-encoding overhead.",
    parameters: Type.Object({
      tools: Type.Array(
        Type.Object({
          name: Type.String({ description: "Tool name" }),
          description: Type.Optional(Type.String({ description: "Tool description" }))
        }),
        { description: "List of tools to pre-cache in memory" }
      )
    }),
    async execute(toolCallId, params, signal) {
      const ports = [8083, 8080];
      let lastError: any = null;

      for (const port of ports) {
        try {
          const url = `http://127.0.0.1:${port}/v1/tools/register`;
          const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tools: params.tools }),
            signal
          });
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status}: ${await res.text()}`);
            continue;
          }
          const data = await res.json();
          return {
            content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
            details: { registered: data.registered_count, status: data.status, port }
          };
        } catch (err: any) {
          lastError = err;
        }
      }

      return {
        content: [{ type: "text", text: `Error registering tools on ports 8083/8080: ${lastError?.message || lastError}` }],
        isError: true
      };
    }
  });

  // 5. CPU CLM Engine Diagnostics & Info
  pi.registerTool({
    name: "cpu_clm_info",
    label: "CPU CLM Status",
    description:
      "Retrieves current CLM model information, vector dimension (2048-dim for CLM-8B vs 1024-dim for Open-JEV), P-core affinity status, and in-memory action cache statistics.",
    parameters: Type.Object({}),
    async execute(toolCallId, params, signal) {
      const ports = [8083, 8080];
      let lastError: any = null;

      for (const port of ports) {
        try {
          const url = `http://127.0.0.1:${port}/v1/clm/info`;
          const res = await fetch(url, { signal });
          if (!res.ok) {
            lastError = new Error(`HTTP ${res.status}: ${await res.text()}`);
            continue;
          }
          const data = await res.json();
          return {
            content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
            details: { model: data.model, is_clm_8b: data.is_clm_8b, port }
          };
        } catch (err: any) {
          lastError = err;
        }
      }

      return {
        content: [{ type: "text", text: `Error querying CLM info on ports 8083/8080: ${lastError?.message || lastError}` }],
        isError: true
      };
    }
  });
}

