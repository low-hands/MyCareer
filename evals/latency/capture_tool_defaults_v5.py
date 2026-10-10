"""Reproduce the Qwen decision configuration used by the v5 latency capture."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from career_agent.evaluation.latency_capture import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--conversations', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.conversations <= 20:
        parser.error('conversations must be between 1 and 20')
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    for suffix in ('BASE_URL', 'API_KEY', 'MODEL'):
        value = os.environ.get('RESUME_ANALYSIS_AGENT_' + suffix)
        if value is None:
            parser.error('configure RESUME_ANALYSIS_AGENT connection before capture')
        os.environ['MAIN_AGENT_' + suffix] = value
    os.environ['MAIN_AGENT_VENDOR'] = 'qwen'
    os.environ['MAIN_AGENT_ENABLE_THINKING'] = 'false'
    capture(args.work_root, conversations=args.conversations)


if __name__ == '__main__':
    main()
