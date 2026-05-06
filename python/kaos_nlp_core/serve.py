"""Run the KAOS MCP server with NLP tools.

Usage:
    # stdio (for Claude Code / Claude Desktop) — single-tenant trust boundary
    kaos-nlp-serve

    # streamable HTTP — REQUIRES KAOS_NLP_HTTP_TOKEN to be set; the value is an
    # operator acknowledgement that the server's tool surface (which can read
    # and write files inside KAOS_NLP_WORKSPACE_ROOT, defaulting to CWD) will
    # be fronted by a reverse proxy doing real authentication. The token is
    # NOT validated against incoming requests by this server; treat it as a
    # gate for "I have a proxy in front of this", not as authentication.
    KAOS_NLP_HTTP_TOKEN=ops-ack kaos-nlp-serve --http --port 8000

    # with debug logging
    kaos-nlp-serve --debug
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kaos-nlp-core MCP server."""
    parser = argparse.ArgumentParser(description="KAOS MCP Server with NLP tools")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    # Gate kaos-core / kaos-mcp imports up-front so a base install of
    # kaos-nlp-core (no siblings) gets the friendly install hint instead of
    # an unfriendly ModuleNotFoundError from settings.py — KaosNlpSettings
    # transitively imports kaos_core.config.ModuleSettings. This must run
    # BEFORE the settings import below; otherwise the chained import error
    # surfaces from inside `from kaos_nlp_core.settings import ...` and the
    # operator never sees the actionable message.
    try:
        from kaos_core import KaosRuntime

        # kaos-mcp ships in 0.1.0a2 (Wave 3). Lazy import resolves at runtime
        # when the optional sibling is installed; ty can't see it statically
        # in the per-module repo where the sibling is absent.
        from kaos_mcp import KaosMCPServer, KaosMCPSettings  # ty: ignore[unresolved-import]
    except ImportError as e:
        print(
            f"kaos-mcp and kaos-core are required for the MCP server: {e}\n"
            "Install with: pip install kaos-core kaos-mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve settings from env (KAOS_NLP_*) + .env file. The HTTP-token gate
    # below uses the typed KaosNlpSettings field (SecretStr) rather than a raw
    # ``os.environ`` read so the value participates in the same redaction +
    # config-dump path as every other module setting.
    from kaos_nlp_core.settings import KaosNlpSettings

    nlp_settings = KaosNlpSettings()

    # F3: --http exposes file-touching tools (kaos-nlp-build-index reads
    # corpus_path and writes output_path inside KAOS_NLP_WORKSPACE_ROOT). This
    # server does not authenticate clients; a deployment that opens the port
    # without a proxy is unsafe by construction. Require the operator to set
    # KAOS_NLP_HTTP_TOKEN as an explicit acknowledgement.
    if args.http and (
        nlp_settings.http_token is None or not nlp_settings.http_token.get_secret_value()
    ):
        print(
            "kaos-nlp-serve --http refuses to start without KAOS_NLP_HTTP_TOKEN.\n"
            "\n"
            "The HTTP transport does not validate incoming requests. The tool "
            "surface (notably kaos-nlp-build-index) reads and writes files "
            "inside KAOS_NLP_WORKSPACE_ROOT (default: CWD). To run safely:\n"
            "  1. Front this process with a reverse proxy that authenticates "
            "callers (mTLS, bearer-token, OAuth, …).\n"
            "  2. Constrain the workspace via KAOS_NLP_WORKSPACE_ROOT to a "
            "directory the proxy users may legitimately touch.\n"
            "  3. Set KAOS_NLP_HTTP_TOKEN=<any-non-empty-string> to confirm "
            "you have done (1) and (2).\n"
            "\n"
            "For local single-tenant use, prefer stdio: `kaos-nlp-serve` "
            "(no --http).",
            file=sys.stderr,
        )
        sys.exit(2)

    from kaos_nlp_core.tools import register_nlp_tools

    # Create runtime and register NLP tools
    runtime = KaosRuntime()
    n_tools = register_nlp_tools(runtime)
    print(f"Registered {n_tools} NLP tools", file=sys.stderr)

    # Configure server
    settings = KaosMCPSettings(
        name="kaos-nlp-server",
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server = KaosMCPServer(runtime=runtime, settings=settings)

    if args.http:
        print(f"Starting HTTP server on {args.host}:{args.port}/mcp", file=sys.stderr)
        server.run_streamable_http()
    else:
        print("Starting stdio server", file=sys.stderr)
        server.run_stdio()


if __name__ == "__main__":
    main()
