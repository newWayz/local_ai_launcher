"""
Interactive terminal prompting and user interaction utilities for local_ai_launcher.
Decoupled from configuration dataclasses and filesystem topology.
"""
import sys
import time
import logging

logger = logging.getLogger("service_launcher")


def prompt_pause(prompt: str = "Press Enter to exit...") -> None:
    """Pause execution and wait for user to press Enter."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") or any(os.environ.get(v) for v in ("CI", "HEADLESS", "NONINTERACTIVE")):
        return
    if sys.stdin and sys.stdin.isatty():
        try:
            input(prompt)
        except (EOFError, KeyboardInterrupt):
            pass


def prompt_choice(prompt: str, default: str = 'n', timeout: int = 5) -> bool:
    """Prompt the user for a y/n choice, with a timeout and non-interactive fallback."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") or any(os.environ.get(v) for v in ("CI", "HEADLESS", "NONINTERACTIVE")):
        return default.lower() == 'y'
    if not (sys.stdin and sys.stdin.isatty()):
        logger.info(f"  -> Non-interactive environment. Auto-deciding '{prompt.strip()}' as default '{default}'")
        return default.lower() == 'y'

    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
        print(prompt, end='', flush=True)
        start_time = time.time()
        input_str = ''
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                if char in ('\r', '\n'):
                    print()
                    choice = input_str.strip().lower()
                    if not choice:
                        choice = default.lower()
                    return choice == 'y'
                elif char == '\x08':  # Backspace
                    if len(input_str) > 0:
                        input_str = input_str[:-1]
                        print("\x08 \x08", end='', flush=True)
                else:
                    input_str += char
                    print(char, end='', flush=True)
            if time.time() - start_time > timeout:
                print(f"\n[Timeout after {timeout}s. Using default: '{default}']")
                return default.lower() == 'y'
            time.sleep(0.05)
    else:
        import select
        print(prompt, end='', flush=True)
        i, _, _ = select.select([sys.stdin], [], [], timeout)
        if i:
            choice = sys.stdin.readline().strip().lower()
            if not choice:
                choice = default.lower()
            return choice == 'y'
        else:
            print(f"\n[Timeout after {timeout}s. Using default: '{default}']")
            return default.lower() == 'y'


def prompt_string(prompt: str, default: str = "", timeout: int = 5) -> str:
    """Prompt the user for a string input, with a timeout and non-interactive fallback."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") or any(os.environ.get(v) for v in ("CI", "HEADLESS", "NONINTERACTIVE")):
        return default
    if not (sys.stdin and sys.stdin.isatty()):
        return default

    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
        print(prompt, end='', flush=True)
        start_time = time.time()
        input_str = ''
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                if char in ('\r', '\n'):
                    print()
                    choice = input_str.strip()
                    if not choice:
                        choice = default
                    return choice
                elif char == '\x08':  # Backspace
                    if len(input_str) > 0:
                        input_str = input_str[:-1]
                        print("\x08 \x08", end='', flush=True)
                else:
                    input_str += char
                    print(char, end='', flush=True)
            if time.time() - start_time > timeout:
                print(f"\n[Timeout after {timeout}s. Using default: '{default}']")
                return default
            time.sleep(0.05)
    else:
        import select
        print(prompt, end='', flush=True)
        i, _, _ = select.select([sys.stdin], [], [], timeout)
        if i:
            choice = sys.stdin.readline().strip()
            if not choice:
                choice = default
            return choice
        else:
            print(f"\n[Timeout after {timeout}s. Using default: '{default}']")
            return default


def fatal_error(msg=None, exit_code=1):
    """Log an issue, pause for user input, and exit."""
    if msg:
        logger.error(msg)
    prompt_pause("Exiting script now. Press enter key to exit >")
    sys.exit(exit_code)


def _resolve_behavior(behavior: str, action_label: str, component_name: str) -> bool:
    """Resolve a 3-way behavior setting ('never_X', 'always_X', 'existing') into a bool."""
    if behavior == f'never_{action_label}':
        logger.info(f"  -> Skipping {component_name} {action_label} as per configuration.")
        return False
    elif behavior == f'always_{action_label}':
        return True
    else:  # 'existing'
        return prompt_choice(f"Do you want to {action_label} {component_name}? (y/n): ", default='n')


__all__ = [
    "prompt_pause",
    "prompt_choice",
    "prompt_string",
    "fatal_error",
    "_resolve_behavior",
]
