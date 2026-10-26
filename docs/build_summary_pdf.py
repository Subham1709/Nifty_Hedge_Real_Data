"""Builds docs/nifty_hedge_real_prices_summary.pdf from the saved results (python docs/build_summary_pdf.py)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "output", ROOT / "figures"
AUTHOR = "Subh"
LINKS = "github.com/Subham1709/Nifty_Hedge_Real_Data · app: subham-nifty-hedge.streamlit.app"
R = json.loads((OUT / "results.json").read_text())
perf = {p: pd.read_csv(OUT / f"performance_{p}.csv", index_col=0) for p in ("real", "model", "original")}
PROPOSED = "VIX-timed 3M put + hybrid monetisation (proposed)"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.4, alignment=TA_LEFT,
                      spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, alignment=TA_LEFT,
                    spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=8, spaceAfter=4)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#52514e"))
cell = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10, spaceAfter=0)


def P(t, st=body):
    return Paragraph(t, st)


def table(rows, widths, bold_row=None):
    t = Table([[P(str(c), cell) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe7")),
             ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#52514e")),
             ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d9d8d4")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#e3eefb")))
    t.setStyle(TableStyle(style))
    return t


def img(name, width=17):
    from PIL import Image as PI
    w, h = PI.open(FIG / name).size
    return Image(str(FIG / name), width=width * cm, height=width * cm * h / w)


real, orig, model = perf["real"], perf["original"], perf["model"]
p, u, po = real.loc[PROPOSED], real.loc["Unhedged"], orig.loc[PROPOSED]
pm = R["premium_model_vs_real"]
r3, r1 = pm["Always 3M 95% put, rolled monthly"], pm["Always 1M 95% put"]
iv = R["iv_daily"]["median"]
lq = R["liquidity"]
xc = R["ex_covid"]
rob = pd.DataFrame(R["robustness"])
hc = rob[(rob.group == "Bid/ask haircut") & (rob.strategy == "timed_hybrid")].set_index("label").CAGR

LABEL = {"Unhedged": "Unhedged", "Always 1M 95% put": "Always 1M 95% put",
         "Always 1M 95/85 put spread": "Always 1M 95/85 put spread",
         "Always 3M 95% put, rolled monthly": "Always 3M 95% put, rolled monthly",
         "VIX-timed 3M put, hold": "VIX-timed 3M put, hold to expiry",
         PROPOSED: "VIX-timed 3M put, hybrid selling (proposed)",
         "VIX-timed 3M put, sell all at VIX 22": "VIX-timed 3M put, sell all at VIX 22"}

rows = [["Strategy", "Return a year, real prices", "Model prices (course project)", "Worst fall, real",
         "Insurance cost a year, real", "Puts bought"]]
for n, lab in LABEL.items():
    r = real.loc[n]
    rows.append([lab, f"{r.CAGR:.1f}%", f"{orig.loc[n].CAGR:.1f}%", f"{r.maxDD:.1f}%", f"{r.net_hedge_cost_pa:.1f}%",
                 f"{int(r.puts_bought)}"])

story = [
    P("Nifty crash insurance, tested on real NSE option prices", h1),
    P(f"{AUTHOR} · {LINKS}", small),
    Spacer(1, 6),
    P("<b>Question.</b> A course project (FDRM, IIT Jodhpur) found that a Nifty 50 portfolio did better with crash "
      "insurance bought only when India VIX was low: a 3-month put 5% below the market, bought on the first trading "
      "day of a month if VIX was below 13, half sold when VIX reached 22 and the rest as the spike faded. The project "
      "had no historical option prices, so every premium came from Black-Scholes with India VIX plus an assumed put "
      "skew. Here the same backtest is rebuilt on "
      f"<b>{R['data']['option_rows']:,} real Nifty option prices</b> from NSE's daily F&amp;O bhavcopies, "
      f"{R['data']['start']} to {R['data']['end']}."),
    P("Results", h2),
    P(f"<b>The model priced the insurance too high.</b> Against the market, its premium was a median "
      f"{r3['median_ratio'] - 1:.0%} too high for 3-month puts and {r1['median_ratio'] - 1:.0%} for 1-month puts, and "
      f"higher in {r3['share_model_dearer']:.0%} of purchases. The market charged an implied volatility of about "
      f"VIX + {iv['iv_3M'] - iv['VIX']:.0f} point for the 3-month put, not the VIX + 3 the model assumed. The gap was "
      "widest when VIX was low, which is when the rule buys."),
    P(f"<b>The rule held up.</b> On real prices it returned <b>{p.CAGR:.1f}% a year</b> against {u.CAGR:.1f}% "
      f"unhedged, with a worst fall of <b>{p.maxDD:.1f}%</b> against {u.maxDD:.1f}% (model prices: {po.CAGR:.1f}%). "
      "Every hedge looks cheaper on real prices, but the ranking is unchanged: buying a put every month still cost "
      "more than it saved, and buying only when VIX was low still beat no hedge. Holding the VIX-timed put to expiry "
      "did even better than the hybrid selling rule, as in the course project."),
    table(rows, [5.6 * cm, 2.2 * cm, 2.6 * cm, 2.1 * cm, 2.4 * cm, 1.6 * cm], bold_row=6),
    P("Jan 2020 to Apr 2026. Real prices include a 2% bid/ask haircut on every trade. A negative insurance cost means "
      "the puts earned more than they cost.", small),
    img("fig2_put_iv_model_vs_market.png", 16.5),
    P("Robustness", h2),
    P(f"The result survives bid/ask haircuts from 0% to 10% ({hc['0%']:.1f}% to {hc['10%']:.1f}% a year), liquidity "
      "cut-offs from 10 to 200 contracts, and buy thresholds from VIX 12 to 14. A 1-month or 2-month put does not "
      "work: the January 2020 put expired before the COVID crash. From July 2020 the rule returned "
      f"{xc[PROPOSED]['CAGR']:.1f}% against {xc['Unhedged']['CAGR']:.1f}% unhedged, so most of the full-period lead "
      "comes from one put bought in January 2020."),
    img("fig4_three_pricings.png", 16.5),
    P("Method", h2),
    P("<b>Contracts.</b> Each put is a real listed contract: the monthly or quarterly expiry nearest the chosen length "
      f"({lq['opens_days_to_expiry']['3M'][0]} to {lq['opens_days_to_expiry']['3M'][2]} days for 3-month puts), and "
      "the strike nearest 5% below Nifty among strikes that traded at least 50 contracts that day. It is held to its "
      "real expiry and settled on the Nifty close."),
    P("<b>Prices.</b> A strike's own NSE closing price is used when at least 50 contracts traded that day. Otherwise "
      "the price comes from that day's implied-volatility smile, fitted to the liquid strikes of the same expiry. "
      "Single-contract prints can be 50% away from neighbouring strikes. "
      f"{lq['trades_by_source'].get('traded', 0)} of {lq['trades']} trades used the option's own price; the median "
      f"put bought traded {lq['opens_median_volume']:,.0f} contracts that day."),
    P("<b>Like-for-like.</b> The engine also prices the same real contracts with the course project's model. That "
      "separates the effect of real prices from the effect of real expiry dates. Period, portfolio, roll dates, "
      "thresholds and metrics are unchanged."),
    P("<b>Limitations.</b> Bhavcopies have end-of-day prices, not bid/ask quotes, so trading cost is an assumption. "
      "Positions are fractions of the portfolio, not whole lots. Dividends are left out. Six years contain one large "
      "crash, so the edge rests on few events.", body),
    P("Code, data download notebook and every number: output/results.json in the repository. For learning and "
      "research, not investment advice.", small),
]
doc = SimpleDocTemplate(str(ROOT / "docs" / "nifty_hedge_real_prices_summary.pdf"), pagesize=A4,
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                        title="Nifty crash insurance on real NSE option prices", author=AUTHOR)
doc.build(story)
print("pdf written")
