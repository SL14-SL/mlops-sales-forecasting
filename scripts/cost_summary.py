import json

from src.monitoring.costs import build_cost_report


def main():
    report = build_cost_report(window_days=7)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()