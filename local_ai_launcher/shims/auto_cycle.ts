// local_ai_launcher/shims/auto_cycle.ts
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import fs from "fs";
import path from "path";
import os from "os";

export default function registerExtension(piContext: ExtensionAPI) {
  piContext.registerCommand("cycle-test", {
    description: "Autonomous implementation and review cycle with caveman and ponytail skills",
    handler: async (target: string, ctx: ExtensionCommandContext) => {
      if (!target || !target.trim()) {
        if (ctx.hasUI) {
          ctx.ui.notify("Usage: /cycle-test <target-or-task-description>", "warning");
        }
        return;
      }

      // Check current git status and branch dynamically
      let originalBranch = "main";
      try {
        const branchRes = await piContext.exec("git", ["rev-parse", "--abbrev-ref", "HEAD"], { cwd: ctx.cwd });
        if (branchRes.code === 0 && branchRes.stdout.trim()) {
          originalBranch = branchRes.stdout.trim();
        }
      } catch {
        // Fallback to main if git rev-parse fails
      }

      // Check if working tree is clean
      try {
        const statusRes = await piContext.exec("git", ["status", "--porcelain"], { cwd: ctx.cwd });
        if (statusRes.code === 0 && statusRes.stdout.trim().length > 0) {
          if (ctx.hasUI) {
            ctx.ui.notify("Warning: Working tree has uncommitted changes. Stash or commit before running /cycle-test.", "warning");
          }
        }
      } catch {
        // Ignore git status error if not a git repo
      }

      const reviewBranch = `review-optimization-${Date.now()}`;

      // Resolve paths to skills
      const agentDir = process.env.PI_CODING_AGENT_DIR || path.join(os.homedir(), ".pi", "agent");
      const gitStorageDir = path.join(agentDir, "git", "github.com");

      // Caveman prompt fallback
      let cavemanPrompt = "";
      const cavemanPaths = [
        path.join(gitStorageDir, "kulapard", "pi-caveman", "skills", "caveman", "SKILL.md"),
        path.join(gitStorageDir, "kulapard", "pi-laconic", "skills", "laconic", "SKILL.md"),
        path.join(agentDir, "skills", "caveman", "SKILL.md"),
      ];
      for (const p of cavemanPaths) {
        if (fs.existsSync(p)) {
          cavemanPrompt = fs.readFileSync(p, "utf8");
          break;
        }
      }

      // Ponytail prompt fallback
      let ponytailPrompt = "";
      const ponytailPaths = [
        path.join(gitStorageDir, "DietrichGebert", "ponytail", "skills", "ponytail", "SKILL.md"),
        path.join(agentDir, "skills", "ponytail", "SKILL.md"),
      ];
      for (const p of ponytailPaths) {
        if (fs.existsSync(p)) {
          ponytailPrompt = fs.readFileSync(p, "utf8");
          break;
        }
      }

      // Detect test command
      let testCmd = "npm";
      let testArgs = ["test"];
      if (fs.existsSync(path.join(ctx.cwd, "pytest.ini")) || fs.existsSync(path.join(ctx.cwd, "pyproject.toml"))) {
        testCmd = "pytest";
        testArgs = [];
      } else if (fs.existsSync(path.join(ctx.cwd, "Cargo.toml"))) {
        testCmd = "cargo";
        testArgs = ["test"];
      }

      // ─────────────────────────────────────────────────────────────────
      // PHASE 1: Rapid Implementation (Caveman Rules Attached)
      // ─────────────────────────────────────────────────────────────────
      if (ctx.hasUI) {
        ctx.ui.notify("Phase 1: Implementing features with Caveman speed...", "info");
      }

      const p1Msg = cavemanPrompt
        ? `${cavemanPrompt}\n\nTask: Implement this target immediately: ${target}`
        : `Task: Implement this target immediately: ${target}`;
      piContext.sendUserMessage(p1Msg);
      await ctx.waitForIdle();

      // ─────────────────────────────────────────────────────────────────
      // PHASE 2: Structural Review (Ponytail Rules Attached on Side-Branch)
      // ─────────────────────────────────────────────────────────────────
      if (ctx.hasUI) {
        ctx.ui.notify(`Phase 2: Creating branch [${reviewBranch}] for structural cleanup...`, "info");
      }
      try {
        await piContext.exec("git", ["checkout", "-b", reviewBranch], { cwd: ctx.cwd });
      } catch {
        // Fallback or notice if git fails
      }

      const p2Msg = ponytailPrompt
        ? `${ponytailPrompt}\n\nTask: Review the code changes just made for ${target}. Apply your decision ladder to strip out any over-engineering.`
        : `Task: Review the code changes just made for ${target}. Apply your decision ladder to strip out any over-engineering.`;
      piContext.sendUserMessage(p2Msg);
      await ctx.waitForIdle();

      // ─────────────────────────────────────────────────────────────────
      // PHASE 3: Automatic Test Gatekeeping
      // ─────────────────────────────────────────────────────────────────
      if (ctx.hasUI) {
        ctx.ui.notify(`Phase 3: Executing test suite (${testCmd} ${testArgs.join(" ")})...`, "info");
      }
      let exitCode = 1;
      try {
        const testExecution = await piContext.exec(testCmd, testArgs, { cwd: ctx.cwd });
        exitCode = testExecution.code;
      } catch {
        exitCode = 1;
      }

      if (exitCode === 0) {
        if (ctx.hasUI) {
          ctx.ui.notify("SUCCESS: Optimization passed! Merging branch...", "info");
        }
        await piContext.exec("git", ["checkout", originalBranch], { cwd: ctx.cwd });
        await piContext.exec("git", ["merge", reviewBranch], { cwd: ctx.cwd });
      } else {
        if (ctx.hasUI) {
          ctx.ui.notify(`CRITICAL: Optimization broke tests! Exit code: ${exitCode}. Aborting auto-merge.`, "error");
        }
        piContext.sendUserMessage(
          `The optimization pass broke the tests with exit code ${exitCode}. Review the codebase on this branch and explain the error.`
        );
      }
    },
  });
}
