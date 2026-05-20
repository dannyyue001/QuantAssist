"""
Sample financial document loader for local testing.
Generates synthetic 10-K/10-Q style excerpts without requiring external data.
"""
import random
import uuid


COMPANIES = ["Apple Inc.", "Microsoft Corp.", "Goldman Sachs", "JPMorgan Chase", "BlackRock"]
SECTIONS = ["Risk Factors", "Management Discussion", "Financial Statements", "Liquidity"]

TEMPLATES = [
    "Revenue for {company} increased {pct}% year-over-year to ${rev}B in fiscal {year}. "
    "Operating margin expanded {margin} basis points to {op_margin}%.",

    "The company faces {risk} risk from rising interest rates. "
    "A 100bps increase in rates would reduce net interest income by approximately ${impact}M.",

    "Cash and cash equivalents stood at ${cash}B as of {date}. "
    "Free cash flow generation was ${fcf}B, representing a {fcf_yield}% FCF yield.",

    "Segment {seg} contributed ${seg_rev}B to total revenue, representing {seg_pct}% of consolidated sales. "
    "Adjusted EBITDA for the segment was ${ebitda}B.",

    "The firm's VaR at the 99th percentile was ${var}M over a one-day horizon. "
    "Stressed VaR under the 2008 financial crisis scenario was ${svar}M.",
]


def generate_document(doc_id: str | None = None) -> dict:
    company = random.choice(COMPANIES)
    section = random.choice(SECTIONS)
    template = random.choice(TEMPLATES)

    text = template.format(
        company=company,
        pct=round(random.uniform(5, 25), 1),
        rev=round(random.uniform(10, 400), 1),
        year=random.choice([2022, 2023, 2024]),
        margin=random.randint(20, 150),
        op_margin=round(random.uniform(15, 45), 1),
        risk=random.choice(["credit", "market", "liquidity", "operational"]),
        impact=random.randint(50, 500),
        cash=round(random.uniform(5, 120), 1),
        date=f"December 31, {random.choice([2023, 2024])}",
        fcf=round(random.uniform(5, 80), 1),
        fcf_yield=round(random.uniform(2, 8), 1),
        seg=random.choice(["Cloud", "Consumer", "Institutional", "Asset Management"]),
        seg_rev=round(random.uniform(5, 80), 1),
        seg_pct=round(random.uniform(10, 40), 1),
        ebitda=round(random.uniform(2, 30), 1),
        var=random.randint(50, 500),
        svar=random.randint(200, 2000),
    )

    return {
        "text": text,
        "metadata": {
            "doc_id": doc_id or str(uuid.uuid4()),
            "company": company,
            "section": section,
            "source": "synthetic_10k",
        },
    }


def generate_dataset(n: int = 100) -> list[dict]:
    return [generate_document() for _ in range(n)]


if __name__ == "__main__":
    import json
    docs = generate_dataset(10)
    for doc in docs:
        print(json.dumps(doc, indent=2))
