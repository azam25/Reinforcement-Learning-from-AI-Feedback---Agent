"""
client.py — Interactive test client for the RLAIF RAG Agent API.

Usage:
    # Make sure the server is running first:
    #   python3 run_server.py
    #
    # Then in another terminal (with venv active):
    #   python3 client.py                        # interactive mode
    #   python3 client.py --ingest path/to.pdf   # ingest a document
    #   python3 client.py --ask "Your question"  # ask a question
    #   python3 client.py --all path/to.pdf      # ingest + ask interactively

Run: python3 client.py --help
"""

import argparse
import json
import sys
import textwrap

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
TIMEOUT  = 120   # seconds (LLM calls can be slow)
# ---------------------------------------------------------------------------


def _print_separator(title: str = ""):
    width = 72
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


def health_check(client: httpx.Client) -> bool:
    """Ping the /health endpoint. Returns True if the server is up."""
    try:
        resp = client.get("/health", timeout=5)
        data = resp.json()
        print(f"  Server: OK  |  Model: {data.get('model', '?')}")
        return True
    except httpx.ConnectError:
        print("  ERROR: Cannot connect to the server at", BASE_URL)
        print("  Start it with:  python3 run_server.py")
        return False


def ingest_document(client: httpx.Client, file_path: str, force_rebuild: bool = False) -> bool:
    """POST /ingest — load and index a PDF."""
    _print_separator("INGEST")
    print(f"  File:  {file_path}")
    print(f"  Force rebuild: {force_rebuild}")

    resp = client.post(
        "/ingest",
        json={"file_path": file_path, "force_rebuild": force_rebuild},
        timeout=TIMEOUT,
    )

    if resp.status_code == 200:
        print(f"  Status: {resp.status_code} OK")
        print(f"  {resp.json()['message']}")
        return True
    else:
        print(f"  Status: {resp.status_code}")
        _print_json(resp.json())
        return False


def ask_question(
    client: httpx.Client,
    question: str,
    rlaif: bool = True,
    verbose: int = 1,
) -> dict | None:
    """POST /ask — answer a question."""
    _print_separator("ASK")
    print(f"  Question: {question}")
    print(f"  RLAIF: {rlaif}  |  Verbose: {verbose}")

    resp = client.post(
        "/ask",
        json={"question": question, "rlaif": rlaif, "verbose": verbose},
        timeout=TIMEOUT,
    )

    data = resp.json()
    if resp.status_code == 200:
        print(f"\n  Has Answer:  {data['has_answer']}")
        print(f"  Iterations:  {data['context_iterations']}")
        print(f"  Sources:     {data['source_count']} chunk(s)")
        _print_separator("ANSWER")
        wrapped = textwrap.fill(data["answer"], width=72, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        return data
    else:
        print(f"  Status: {resp.status_code}")
        _print_json(data)
        return None


def interactive_session(client: httpx.Client):
    """REPL: ask multiple questions against the already-loaded document."""
    _print_separator("INTERACTIVE SESSION")
    print("  Type your question and press Enter.")
    print("  Commands:  /rlaif on|off   /verbose 0|1|2   /quit")
    print()

    rlaif   = True
    verbose = 1

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not line:
            continue
        if line == "/quit":
            print("  Bye!")
            break
        if line.startswith("/rlaif "):
            rlaif = line.split()[1].lower() == "on"
            print(f"  RLAIF set to: {rlaif}")
            continue
        if line.startswith("/verbose "):
            verbose = int(line.split()[1])
            print(f"  Verbose set to: {verbose}")
            continue

        ask_question(client, line, rlaif=rlaif, verbose=verbose)


def _print_json(data: dict):
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Test client for the RLAIF RAG Agent API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 client.py                            # health check
          python3 client.py --ingest /path/to/doc.pdf  # ingest document
          python3 client.py --ask "What is X?"         # single question
          python3 client.py --all /path/to/doc.pdf     # ingest + interactive Q&A
          python3 client.py --ingest /path/to/doc.pdf --ask "What is X?" --no-rlaif
        """),
    )
    parser.add_argument("--base-url", default=BASE_URL, help=f"API base URL (default: {BASE_URL})")
    parser.add_argument("--ingest",   metavar="FILE",  help="PDF file to ingest")
    parser.add_argument("--ask",      metavar="Q",     help="Single question to ask")
    parser.add_argument("--all",      metavar="FILE",  help="Ingest FILE then start interactive Q&A")
    parser.add_argument("--force-rebuild", action="store_true", help="Force re-embedding even if index exists")
    parser.add_argument("--no-rlaif", action="store_true", help="Disable RLAIF self-evaluation")
    parser.add_argument("--verbose",  type=int, default=1, choices=[0, 1, 2], help="Verbosity (0-2)")
    args = parser.parse_args()

    BASE_URL = args.base_url.rstrip("/")
    rlaif    = not args.no_rlaif

    with httpx.Client(base_url=BASE_URL) as client:
        _print_separator("RLAIF RAG Agent — Test Client")
        if not health_check(client):
            sys.exit(1)

        if args.all:
            ok = ingest_document(client, args.all, force_rebuild=args.force_rebuild)
            if ok:
                interactive_session(client)

        elif args.ingest and args.ask:
            ok = ingest_document(client, args.ingest, force_rebuild=args.force_rebuild)
            if ok:
                ask_question(client, args.ask, rlaif=rlaif, verbose=args.verbose)

        elif args.ingest:
            ingest_document(client, args.ingest, force_rebuild=args.force_rebuild)

        elif args.ask:
            ask_question(client, args.ask, rlaif=rlaif, verbose=args.verbose)

        else:
            # No explicit action — just show health and usage hint
            print()
            print("  No action specified. Run with --help for options.")
            print(f"  API Docs: {BASE_URL}/docs")

        _print_separator()


if __name__ == "__main__":
    main()
