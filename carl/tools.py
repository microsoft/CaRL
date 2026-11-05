"""Explicit tool setup for all public entry points."""
from carl.sandbox.trusted import TrustedExecutor


def make_executor(args):
    search = None
    if args.index:
        from carl.retrieval.search import BM25Search
        search = BM25Search(args.index)
    if getattr(args, "container_image", None):
        return TrustedExecutor(search=search, container_image=args.container_image)
    return TrustedExecutor(allow_trusted_execution=args.allow_trusted_execution, search=search)
