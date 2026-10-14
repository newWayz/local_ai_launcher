import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * Autonomous Loop Recovery & Escalation Extension for Pi.
 *
 * Intercepts loop halt notices emitted by the local_ai_launcher gateway,
 * autonomously re-prompts the model with anti-attractor instructions up to
 * PI_LOOP_RECOVERY_MAX_RETRIES times (default: 3, user-configurable in config.py / launcher.toml),
 * and escalates to context compaction and slot-0 KV cache eviction when the threshold is reached.
 */
export default function (pi: ExtensionAPI) {
  const sessionLoops = new Map<string, number>();

  pi.on("turn_end", async (event, ctx) => {
    let text = "";
    if (typeof event.message?.content === "string") {
      text = event.message.content;
    } else if (Array.isArray(event.message?.content)) {
      text = event.message.content
        .map((c: any) => (typeof c === "string" ? c : c.text || ""))
        .join("\n");
    }

    const isLoopHalt =
      text.includes("repetition loop detected by local_ai_launcher") ||
      text.includes("repetitive reasoning loop");

    if (!isLoopHalt) return;

    const rawRetries = parseInt(process.env.PI_LOOP_RECOVERY_MAX_RETRIES || "3", 10);
    const maxRetries = isNaN(rawRetries) ? 3 : Math.max(0, rawRetries);
    const sessionId = ctx.sessionManager.getSessionId() || "default";
    const count = (sessionLoops.get(sessionId) || 0) + 1;
    sessionLoops.set(sessionId, count);

    if (count < maxRetries) {
      if (ctx.hasUI) {
        ctx.ui.notify(
          `[Loop Recovery] Intercepted loop #${count}/${maxRetries}. Auto-continuing with alternate hypothesis...`,
          "warning"
        );
      }
      pi.sendUserMessage(
        "The previous reasoning path entered a repetition loop and was halted by the local AI gateway. Discard that hypothesis, error path, or command completely. Propose a distinctly different alternative approach or run a diagnostic test to gather new evidence.",
        { deliverAs: "followUp" }
      );
      return;
    }

    // Escalation: max retries reached -> compact context and erase llama-server KV slot
    if (ctx.hasUI) {
      ctx.ui.notify(
        `[Loop Recovery] Reached max loop threshold (${count}/${maxRetries}). Compacting context & resetting cache...`,
        "error"
      );
    }

    ctx.compact({
      customInstructions:
        "The model was stuck in repetitive reasoning loops. Strip out all repetitive loops, failed attempts, and redundant error logs. Retain only confirmed facts, active project goals, and verified file state.",
      onComplete: async () => {
        try {
          await fetch("http://127.0.0.1:9931/slots/0?action=erase", { method: "POST" });
        } catch {
          // llama-server may be on another port or slot
        }
        sessionLoops.set(sessionId, 0);
        pi.sendUserMessage(
          "Context has been compacted and the KV cache refreshed after persistent repetition loops. State the primary objective and execute the next clean step.",
          { deliverAs: "followUp" }
        );
      },
      onError: async (err) => {
        if (ctx.hasUI) {
          ctx.ui.notify(`[Loop Recovery] Compaction notice: ${err.message}`, "error");
        }
        sessionLoops.set(sessionId, 0);
        pi.sendUserMessage(
          "Resetting turn after persistent loops. Proceed with an alternative approach.",
          { deliverAs: "followUp" }
        );
      },
    });
  });

  pi.on("session_shutdown", () => {
    sessionLoops.clear();
  });
}
