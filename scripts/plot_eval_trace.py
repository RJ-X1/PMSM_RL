"""Backward-compatible wrapper for the canonical episode-trace plotter."""

from scripts.plot_episode_trace import build_arg_parser, create_episode_trace_plot, main

__all__ = ["build_arg_parser", "create_episode_trace_plot", "main"]


if __name__ == "__main__":
    main()
