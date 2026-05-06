import json
import math
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"

EXCEL_PATH = DATA_DIR / "RQ_generator_dataset.xlsx"
CATEGORY_SCHEMA_PATH = DATA_DIR / "category_schema.json"

COL_TITLE = "title"
COL_SUMMARY_PRIMARY = "theoretical_summary"
COL_SUMMARY_FALLBACK = "results"
COL_Y = "Dependent_Variable_Y"
COL_TOPIC_L1 = "Topic_L1"
COL_TOPIC_L2 = "Topic_L2"
COL_Y_CATEGORY = "Y_Category"

MODEL_NAME = "gpt-5.4-nano"


# =========================
# Helpers
# =========================
def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def normalize_y(y: str) -> str:
    if y is None:
        return ""
    y = str(y).strip().lower()
    y = y.replace("−", "-")  # U+2212
    y = y.replace("–", "-")
    y = y.replace("—", "-")
    
    y = " ".join(y.split())

    synonym_map = {
        "claim counts": "claim count",
        "number of claims": "claim count",
        "claims": "insurance claims",
        "mortality rates": "mortality rate",
        "mortality trends": "mortality trend",
        "premiums": "premium",
        "insurance premiums": "insurance premium",
        "solvency capital requirement (scr)": "solvency capital requirement",
        "solvency capital requirements": "solvency capital requirement",
        "value-at-risk": "var",
        "value-at-risk (var)": "var",
        "probability of lifetime ruin": "lifetime ruin probability",
        "ruin probability": "ruin probability",
        "time to ruin": "time to ruin",
    }
    return synonym_map.get(y, y)


def load_categories(schema_path: Path) -> list[str]:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"{schema_path} 파일이 비어 있습니다.")

    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{schema_path}가 올바른 JSON 형식이 아닙니다: {e}") from e

    if isinstance(schema, list):
        return schema

    if isinstance(schema, dict):
        if "top_categories" in schema and isinstance(schema["top_categories"], list):
            return schema["top_categories"]
        if "categories" in schema and isinstance(schema["categories"], list):
            return schema["categories"]

    raise ValueError("category_schema.json 형식을 확인해야 합니다.")


def build_system_prompt(categories: list[str]) -> str:
    cat_text = "\n".join(f"- {c}" for c in categories)

    return f"""
You classify dependent variables from academic papers into exactly one category.

Your task:
1. Read the dependent variable Y.
2. Use title, theoretical summary, Topic_L1, and Topic_L2 as context.
3. Assign exactly one category from the allowed list.
4. Return JSON only.

Allowed categories:
{cat_text}

Output JSON format:
{{
  "y_category": "exactly one category from the allowed list",
  "reason": "brief explanation"
}}

Rules:
- Always assign the most semantically appropriate category.
- Use "Socioeconomic & Policy Outcomes" for external macroeconomic variables, labor market indicators, policy impacts, welfare-related outcomes, or broader social outcomes.
- Use "Other" only if the variable is malformed, unclear, purely methodological/model-performance oriented, or does not clearly fit the domain categories.
- For ruin-related outcomes (e.g. ruin probability, lifetime ruin, time to ruin), prefer "Capital, Solvency & Risk Capital" unless the context clearly indicates a different interpretation.
- y_category must exactly match one allowed category.
- Return JSON only.
- Be consistent and conservative.
""".strip()


def get_summary_text(row: pd.Series) -> str:
    primary = row.get(COL_SUMMARY_PRIMARY, "")
    if not is_empty(primary):
        return str(primary)
    fallback = row.get(COL_SUMMARY_FALLBACK, "")
    if not is_empty(fallback):
        return str(fallback)
    return ""

def heuristic_category(normalized_y: str) -> str:
    y = normalized_y.lower()

    # 0. clearly malformed / methodological / model-performance -> Other
    if (
        "empirical data of" in y
        or "response variable" in y
        or "regression outcomes" in y
        or "model parameters" in y
        or "parameter estimates" in y
        or "distribution model" in y
        or "matrix validity" in y
        or "heatmap reconstruction accuracy" in y
        or "classification accuracy" in y
        or "prediction accuracy" in y
        or "predictive performance" in y
        or "predictive power" in y
        or "forecast accuracy" in y
        or "generalization error" in y
        or "mean squared error" in y
        or "mse" in y
        or "root mean squared error" in y
        or "rmse" in y
        or "model selection accuracy" in y
        or "estimation error" in y
        or "approximation error" in y
        or "computation time" in y
        or "computational time" in y
        or "conditional expectation" in y
        or "distribution function" in y
        or "quantile-regression" in y
        or "point and interval estimator" in y
        or "smoothing results" in y
        or "proxy model accuracy" in y
        or "model error" in y
        or "parameter risk" in y
        or "bias" in y
        or "fit" in y
        or y == "mean"
        or "spectral measure" in y
        or "conditional distribution" in y
        or "limit distribution" in y
        or "projection density" in y
        or "density estimates" in y
    ):
        return "Other"

    # 1. socioeconomic / policy outcomes
    if (
        "unemployment" in y
        or "employment" in y
        or "welfare" in y
        or "economic growth" in y
        or "inflation" in y
        or "epidemic" in y
        or "pandemic" in y
        or "covid" in y
        or "policy" in y
        or "public goods" in y
        or "social insurance" in y
        or "social security" in y
        or "poverty" in y
        or "income inequality" in y
        or "household income" in y
        or "food sufficiency" in y
        or "quality of life" in y
        or "domestic consumption" in y
        or "exports" in y
        or "co2 emissions" in y
        or "environmental violations" in y
        or "pollution" in y
        or "vaccination" in y
        or "hospital admissions" in y
        or "patient survival" in y
        or "health expenditures" in y
        or "healthcare costs" in y
        or "medical expenditures" in y
        or "medical service utilization" in y
        or "physician visits" in y
        or "emergency department visits" in y
        or "health care utilization" in y
        or "pension fairness" in y
        or "fairness in insurance" in y
        or "living standards continuity" in y
        or "farm income compensation" in y
        or "farm income" in y
        or "social capital" in y
        or "worker wealth transfer" in y
        or "disaster losses" in y
        or "economic losses" in y
        or "financial and mental difficulties" in y
        or "public funding" in y
        or "voter support" in y
        or "student engagement" in y
        or "student performance" in y
        or "student learning" in y
        or "education effectiveness" in y
        or "knowledge of" in y
    ):
        return "Socioeconomic & Policy Outcomes"

    # 2. ruin / solvency / capital
    if (
        "ruin probability" in y
        or "lifetime ruin" in y
        or "retirement ruin" in y
        or "time to ruin" in y
        or "deficit at ruin" in y
        or "maximum severity of ruin" in y
        or "gerber-shiu" in y
        or "expected discounted penalty" in y
        or "solvency" in y
        or "capital requirement" in y
        or "risk capital" in y
        or "economic capital" in y
        or "available capital" in y
        or "capital charge" in y
        or "capital adequacy ratio" in y
        or "capital injection" in y
        or "capital estimate" in y
        or "economic own funds" in y
        or "insolvency" in y
        or "default probability" in y
        or "probability of default" in y
        or "bankruptcy probability" in y
        or "bankruptcy risk" in y
        or "bankruptcy costs" in y
        or "default risk" in y
        or "surplus stability" in y
        or "maximum surplus before ruin" in y
        or "surplus process" in y
        or "insurer surplus" in y
        or "terminal surplus" in y
        or "balance sheet stability" in y
        or "life insurer's risk situation" in y
        or "insurer's solvency" in y
        or "default losses" in y
        or "firm distress" in y
        or "financial distress" in y
        or "insurer default" in y   
    ):
        return "Capital, Solvency & Risk Capital"

    # 3. mortality / longevity / demographic
    if (
        "mortality" in y
        or "longevity" in y
        or "life expectancy" in y
        or "survival probability" in y
        or "survival probabilities" in y
        or "death rate" in y
        or "death rates" in y
        or "death counts" in y
        or "number of deaths" in y
        or "age at death" in y
        or "human life length" in y
        or "lifespan" in y
        or "force of mortality" in y
        or "life-table" in y
        or "loaded life tables" in y
        or "biometric indices" in y
        or "health-state transition" in y
        or "disability transition" in y
        or "disability and recovery rates" in y
        or "critical illness diagnosis rates" in y
        or "long-term care state transitions" in y
        or "dependent lifetimes" in y
        or "censored residual lifetimes" in y
        or "joint lifetimes" in y
        or "insured deaths" in y
        or "excess deaths" in y
        or "death probabilities" in y
        or "age distribution of deaths" in y
        or "old-age care prevalence" in y
        or "adl limitations" in y
        or "formal ltc usage" in y
        or "chronic disability" in y
        or "disability probabilities" in y
        or "disability status" in y
        or "disability prevalence" in y
    ):
        return "Mortality, Longevity & Demographic Risk"

    # 4. reserves / liabilities / technical provisions
    if (
        "reserve" in y
        or "reserves" in y
        or "liabilit" in y
        or "technical provision" in y
        or "best estimate" in y
        or "ibnr" in y
        or "outstanding claim" in y
        or "unpaid claim" in y
        or "unpaid loss" in y
        or "claim liabilities" in y
        or "loss development factor" in y
        or "future discretionary benefits" in y
        or "fdb" in y
        or "future benefits" in y
        or "policy benefits" in y
        or "pension liabilities" in y
    ):
        return "Insurance Liabilities, Reserves & Actuarial Quantities"

    # 5. claims / losses / insurance payments
    if (
        "claim count" in y
        or "number of claims" in y
        or "claim frequency" in y
        or "claim probability" in y
        or "claim rate" in y
        or "claim rates" in y
        or "claim amount" in y
        or "claim severity" in y
        or "claim cost" in y
        or "claim costs" in y
        or "claim payment" in y
        or "claims payment" in y
        or "claim filing" in y
        or "claims adjudication" in y
        or "insurance claims" in y
        or "future insurance claims" in y
        or "ultimate loss" in y
        or "ultimate cost" in y
        or "total loss" in y
        or "total losses" in y
        or "aggregate loss" in y
        or "aggregate claims" in y
        or "loss amount" in y
        or "loss amounts" in y
        or "loss severity" in y
        or "loss frequency" in y
        or "loss frequencies" in y
        or "loss cost" in y
        or "loss costs" in y
        or "incurred loss" in y
        or "property damage" in y
        or "auto bodily injury" in y
        or "workers' compensation" in y
        or "disability insurance claims" in y
        or "catastrophe loss" in y
        or "catastrophic loss" in y
        or "hurricane loss" in y
        or "flood loss" in y
        or "earthquake" in y
        or "wildfire" in y
        or "tornado damage" in y
        or "drought damage" in y
        or "crop yields" in y
        or "frost losses" in y
        or "livestock mortality risk" in y
        or "compensation payout" in y
        or "payment amounts" in y
        or "incremental payments" in y
        or "loss payments" in y
        or "insurance payouts" in y
        or "reported loss" in y
        or "economic loss" in y
        or "annual loss" in y
        or "future losses" in y
        or "incremental losses" in y
        or "loss data" in y
        or "insurance loss data" in y
        or "claim occurrence" in y
        or "accident count" in y
        or "accident frequency" in y
        or "accident rate" in y
        or "accident risk" in y
        or "accidents" in y
        or "fire insurance claim data" in y
        or "insurance claim records" in y
        or "comprehensive motor insurance losses" in y
        or "motor insurance losses" in y
        or "auto claims costs" in y
        or "repair costs" in y
    ):
        return "Insurance Claims & Loss Outcomes"

    # 6. pricing / premiums / valuation
    if (
        "premium" in y
        or "premiums" in y
        or "insurance price" in y
        or "price of" in y
        or "pricing" in y
        or "annuity price" in y
        or "annuity payout" in y
        or "annuity value" in y
        or "life insurance product price" in y
        or "life insurance premiums" in y
        or "term life insurance prices" in y
        or "contract value" in y
        or "policy value" in y
        or "product value" in y
        or "fair value" in y
        or "market value" in y
        or "valuation" in y
        or "embedded value" in y
        or "option value" in y
        or "option prices" in y
        or "guarantee value" in y
        or "guarantee cost" in y
        or "guarantee charge" in y
        or "gmm" in y
        or "gmwb" in y
        or "gmdb" in y
        or "glwb" in y
        or "cat bond price" in y
        or "cat bond premium" in y
        or "catastrophe bond price" in y
        or "catastrophe bond spread" in y
        or "mortality bond price" in y
        or "longevity bond" in y
        or "survivor derivative" in y
        or "q-forward" in y
        or "swap spread" in y
        or "warranty price" in y
        or "reverse mortgage price" in y
        or "mortgage insurance premium" in y
        or "pension buy-out price" in y
        or "buyout price" in y
        or "commercial premium" in y
        or "loaded premium" in y
        or "fair fee" in y
        or "fees" in y
        or "benefit present value" in y
        or "present value of future payments" in y
        or "death benefit value" in y
        or "living benefits" in y
        or "benefit stream" in y
        or "benefit withdrawal rate" in y
        or "price levels" in y
        or "contract price" in y
        or "settlement amount" in y
        or "insurance rates" in y
        or "risk margin" in y
        or "summary cost indicator" in y
        or "cost functional" in y
    ):
        return "Pricing, Premiums & Product Valuation"

    # 7. risk measures / tail / extreme value
    if (
        "tail" in y
        or "var" in y
        or "value-at-risk" in y
        or "value at risk" in y
        or "cvar" in y
        or "conditional value at risk" in y
        or "conditional tail expectation" in y
        or "cte" in y
        or "expected shortfall" in y
        or "tail conditional expectation" in y
        or "risk measure" in y
        or "risk measures" in y
        or "distortion risk" in y
        or "haezendonck-goovaerts" in y
        or "h-g risk" in y

        # EVT / extremes
        or "extreme quantile" in y
        or "extreme value" in y
        or "extremal" in y
        or "extreme precipitation" in y
        or "extreme wind" in y
        or "storm peak" in y
        or "river flow" in y
        or "rainfall" in y
        or "burned area" in y
        or "surface temperature extremes" in y
        or "temperature anomalies" in y
        or "max-stable" in y
        or "max stable" in y
        or "spectral measure" in y
        or "exponent measure" in y
        or "angular dependence" in y
        or "marginal extremes" in y
        or "conditional extremes" in y
        or "multivariate extremes" in y
        or "joint extremes" in y
        or "extreme dependence" in y
        or "extreme event" in y

        # exceedance / threshold
        or "exceedance" in y
        or "threshold exceedance" in y
        or "threshold" in y
        or "high threshold" in y
        or "right endpoint" in y

        # maxima / records
        or "maxima" in y
        or "maximum values" in y
        or "record values" in y
        or "annual maximum" in y
        or "maximum temperature" in y
        or "maximum jump length" in y

        # tail dependence / copula
        or "tail index" in y
        or "tail dependence" in y
        or "copula" in y
        or "copula approximation" in y
        or "dependence parameter" in y
        or "dependence measure" in y
        or "positive quadrant dependence" in y
        or "joint survival probability" in y
        or "joint exceedance probability" in y
        or "probability of high extremes" in y
        or "probability of exceeding high threshold" in y
        or "risk concentration" in y

        # EVT estimators / processes
        or "extremogram" in y
        or "pickands" in y
        or "brown-resnick" in y
        or "generalized pareto" in y
        or "hill estimator" in y
        or "heavy-tailed" in y
        or "heavy tailed" in y
        or "conditional quantile" in y
        or "quantile estimates" in y
        or "l_p-quantiles" in y
        or "occupation times" in y
        or "stochastic process paths" in y
        or "supremum" in y
        or "infimum" in y
        or "crossings" in y
        or "first event time" in y
    ):
        return "Risk Measures, Tail Risk & Extreme Value Quantities"

    # 8. investment / portfolio / financial market
    if (
        "stock return" in y
        or "stock returns" in y
        or "stock price" in y
        or "stock prices" in y
        or "equity return" in y
        or "equity returns" in y
        or "share return" in y
        or "share value" in y
        or "bond price" in y
        or "bond prices" in y
        or "bond index" in y
        or "yield curve" in y
        or "interest rate" in y
        or "libor" in y
        or "swap rate" in y
        or "cds spread" in y
        or "credit default swap" in y
        or "credit rating" in y
        or "sovereign credit" in y
        or "exchange rate" in y
        or "currency" in y
        or "bitcoin" in y
        or "cryptocurrency" in y
        or "portfolio" in y
        or "asset allocation" in y
        or "optimal investment" in y
        or "investment choice" in y
        or "investment efficiency" in y
        or "fund performance" in y
        or "fund returns" in y
        or "portfolio performance" in y
        or "portfolio returns" in y
        or "hedging" in y
        or "hedge" in y
        or "basis risk" in y
        or "market risk" in y
        or "market efficiency" in y
        or "market reaction" in y
        or "bid-ask spread" in y
        or "liquidity" in y
        or "volatility" in y
        or "beta" in y
        or "sharpe ratio" in y
        or "return distribution" in y
        or "futures" in y
        or "commodity returns" in y
        or "future yields" in y
        or "discount curve" in y
        or "capital allocation" in y
        or "future net asset value" in y
        or "net present value" in y
        or "npv" in y
        or "market capitalization" in y
        or "equity prices" in y
        or "log prices" in y
        or "asset returns" in y
        or "aggregate stochastic returns" in y
        or "average annual returns" in y
        or "risk-return profile" in y
        or "risk and return" in y
        or "drawdown" in y
        or "wealth at death" in y
        or "terminal wealth" in y
        or "final wealth" in y
        or "wealth accumulation" in y
        or "fund wealth" in y
        or "household capital" in y
    ):
        return "Investment, Portfolio & Asset Performance"

    # 9. firm / insurer / bank performance
    if (
        "firm value" in y
        or "company value" in y
        or "insurer value" in y
        or "shareholder value" in y
        or "shareholder wealth" in y
        or "shareholder profits" in y
        or "profitability" in y
        or "profit" in y
        or "profits" in y
        or "roa" in y
        or "roe" in y
        or "return on assets" in y
        or "return on equity" in y
        or "tobin" in y
        or "firm performance" in y
        or "insurer performance" in y
        or "bank performance" in y
        or "financial performance" in y
        or "operating performance" in y
        or "underwriting performance" in y
        or "underwriting gain" in y
        or "combined ratio" in y
        or "loss ratio" in y
        or "loss ratios" in y
        or "market share" in y
        or "firm growth" in y
        or "growth rate" in y
        or "capital structure" in y
        or "leverage" in y
        or "cash holdings" in y
        or "cost of equity" in y
        or "cost of debt" in y
        or "cost of funds" in y
        or "earnings" in y
        or "dividend payout" in y
        or "dividend payments" in y
        or "abnormal stock returns" in y
        or "cumulative abnormal returns" in y
    ):
        return "Firm Performance & Financial Outcomes"

    # 10. demand / choice / policyholder behavior
    if (
        "insurance demand" in y
        or "insurance consumption" in y
        or "insurance purchase" in y
        or "insurance purchases" in y
        or "insurance uptake" in y
        or "take-up" in y
        or "uptake rate" in y
        or "insurance enrollment" in y
        or "insurance coverage" in y
        or "flood insurance coverage" in y
        or "health insurance coverage" in y
        or "long-term care insurance purchase" in y
        or "demand for ltc" in y
        or "micro-insurance demand" in y
        or "corporate insurance purchase" in y
        or "private insurance adoption" in y
        or "insurance plan choice" in y
        or "health plan choice" in y
        or "annuitization" in y
        or "life insurance purchase" in y
        or "life insurance holdings" in y
        or "surrender" in y
        or "lapse" in y
        or "policy lapse" in y
        or "policyholder retention" in y
        or "renewal" in y
        or "churn" in y
        or "customer retention" in y
        or "willingness to pay" in y
        or "willingness-to-pay" in y
        or "willingness-to-accept" in y
        or "wta/wtp" in y
        or "consumer evaluation" in y
        or "purchase intention" in y
        or "trust in insurance" in y
        or "policyholder utility" in y
        or "policyholder welfare" in y
        or "policyholder participation" in y
        or "total withdrawals" in y
        or "withdrawal intentions" in y
        or "withdrawal strategy" in y
        or "surrender behaviour" in y
        or "surrender behavior" in y
        or "surrender decision" in y
        or "lapse behavior" in y
        or "life insurance lapsation" in y
        or "annuity provider choice" in y
        or "pension plan choice" in y
        or "plan switching" in y
        or "reenrollment likelihood" in y
        or "claim filing decision" in y
        or "acceptance rate" in y
        or "customer attrition" in y
        or "consumer behavior" in y
        or "customer preferences" in y
    ):
        return "Insurance Demand, Choice & Policyholder Behavior"

    # 11. contract design / reinsurance / optimization
    if (
        "optimal insurance contract" in y
        or "optimal reinsurance" in y
        or "reinsurance demand" in y
        or "reinsurance purchase" in y
        or "reinsurance premium" in y
        or "reinsurance contract" in y
        or "reinsurance strategy" in y
        or "reinsurance share" in y
        or "retention level" in y
        or "retained loss" in y
        or "deductible" in y
        or "indemnity" in y
        or "indemnities" in y
        or "ceded loss" in y
        or "risk sharing" in y
        or "risk allocation" in y
        or "risk retention" in y
        or "dividend strategy" in y
        or "optimal dividend" in y
        or "optimal asset allocation" in y
        or "optimal portfolio choice" in y
        or "consumption, investment, life insurance" in y
        or "contract choice" in y
        or "contract design" in y
        or "contract interaction" in y
        or "bonus-malus" in y
        or "surplus participation" in y
        or "profit share" in y
        or "premium rule" in y
        or "premium principle" in y
    ):
        return "Contract Design, Strategy & Optimization"

    # 12. operational / cyber / systemic / fraud / governance-risk
    if (
        "operational risk" in y
        or "operational loss" in y
        or "operational losses" in y
        or "operational resilience" in y
        or "operational performance" in y
        or "operational risk capital" in y
        or "cyber risk" in y
        or "cyber loss" in y
        or "data breach" in y
        or "breach frequency" in y
        or "cybersecurity" in y
        or "fraud" in y
        or "fraudulent" in y
        or "misrepresentation" in y
        or "unauthorized trading" in y
        or "sanctions screening" in y
        or "internal audit" in y
        or "internal control" in y
        or "audit report lag" in y
        or "operational risk disclosure" in y
        or "risk disclosure" in y
        or "risk management practices" in y
        or "erm" in y
        or "risk culture" in y
        or "systemic risk" in y
        or "contagion risk" in y
        or "systemic vulnerabilities" in y
        or "connectedness" in y
        or "banking crisis" in y
        or "banking stability" in y
        or "financial stability" in y
        or "bank failures" in y
        or "financial misconduct" in y
        or "compliance risk" in y
        or "regulatory scrutiny" in y
    ):
        return "Operational Risk, Cyber Risk & Systemic Risk"

    # 13. behavioral / preferences / decision-making
    if (
        "risk aversion" in y
        or "risk attitude" in y
        or "risk attitudes" in y
        or "risk tolerance" in y
        or "risk preference" in y
        or "risk preferences" in y
        or "risk taking" in y
        or "risk-taking" in y
        or "financial risk taking" in y
        or "ambiguity" in y
        or "ambiguity aversion" in y
        or "ambiguity preference" in y
        or "loss aversion" in y
        or "time preference" in y
        or "time preferences" in y
        or "discount rate" in y
        or "time discounting" in y
        or "present bias" in y
        or "choice behavior" in y
        or "choice likelihood" in y
        or "choice frequency" in y
        or "decision-making" in y
        or "decision making" in y
        or "decision time" in y
        or "decision type" in y
        or "risky decision" in y
        or "lottery" in y
        or "gamble" in y
        or "certainty equivalent" in y
        or "preference reversal" in y
        or "expected utility violation" in y
        or "allais" in y
        or "sure-thing principle" in y
        or "belief accuracy" in y
        or "belief formation" in y
        or "subjective probability" in y
        or "probability weighting" in y
        or "value of statistical life" in y
        or "vsl" in y
        or "self-protection" in y
        or "self-insurance" in y
        or "protective measures" in y
        or "protective actions" in y
        or "choice" == y
        or "choice optimality" in y
        or "subject behavior" in y
        or "regret" in y
        or "complexity aversion" in y
        or "risk compensation" in y
        or "self-confidence" in y
        or "reported happiness" in y
        or "attention to prices" in y
        or "beliefs" in y
        or "information acquisition" in y
        or "information accuracy" in y
        or "preference" in y
        or "preferences" in y
        or "utility" == y
        or "rationality" in y
    ):
        return "Behavioral, Preferences & Decision-Making"

    return ""

def classify_y_once(
    client: OpenAI,
    categories: list[str],
    dependent_y: str,
    title: str = "",
    theoretical_summary: str = "",
    topic_l1: str = "",
    topic_l2: str = "",
    normalized_y: str = "",
) -> dict:
    system_prompt = build_system_prompt(categories)

    payload = {
        "dependent_variable_y": dependent_y,
        "normalized_y": normalized_y,
        "title": title,
        "theoretical_summary": theoretical_summary,
        "topic_l1": topic_l1,
        "topic_l2": topic_l2,
    }

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    text = response.output_text.strip()
    result = json.loads(text)

    if "y_category" not in result:
        raise ValueError(f"Invalid response: {result}")

    if result["y_category"] not in categories:
        raise ValueError(f"Category not allowed: {result['y_category']}")

    return result


# =========================
# Main
# =========================
def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    if not CATEGORY_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Category schema file not found: {CATEGORY_SCHEMA_PATH}")

    categories = load_categories(CATEGORY_SCHEMA_PATH)

    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

    if COL_Y_CATEGORY not in df.columns:
        df[COL_Y_CATEGORY] = pd.Series([""] * len(df), dtype="object")
    else:
        df[COL_Y_CATEGORY] = df[COL_Y_CATEGORY].astype("object")

    # 처리 대상: Y는 있고, Y Category는 비어 있는 row
    target_mask = df[COL_Y].notna() & df[COL_Y].astype(str).str.strip().ne("") & (
        df[COL_Y_CATEGORY].isna() | df[COL_Y_CATEGORY].astype(str).str.strip().eq("")
    )

    target_df = df[target_mask].copy()

    if target_df.empty:
        print("No new rows to process.")
        return

    # 같은 Y는 한 번만 GPT 호출
    target_df["_normalized_y"] = target_df[COL_Y].apply(normalize_y)

    unique_y_rows = (
        target_df.sort_index()
        .drop_duplicates(subset=["_normalized_y"], keep="first")
        .copy()
    )

    print(f"Rows to update: {len(target_df)}")
    print(f"Unique Y values to classify: {len(unique_y_rows)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 없거나 로드되지 않았습니다.")

    client = OpenAI(api_key=api_key)

    y_to_category: dict[str, str] = {}
    success_count = 0

    for _, row in unique_y_rows.iterrows():
        original_y = "" if is_empty(row[COL_Y]) else str(row[COL_Y]).strip()
        normalized_y = row["_normalized_y"]

        # 1) heuristic override 먼저 적용
        override_category = heuristic_category(normalized_y)
        if override_category:
            if override_category not in categories:
                raise ValueError(f"Override category not found in category_schema.json: {override_category}")
            y_to_category[normalized_y] = override_category
            success_count += 1
            print(f"[RULE] {original_y} -> {override_category}")
            continue

        title = "" if is_empty(row.get(COL_TITLE, "")) else str(row.get(COL_TITLE, ""))
        summary = get_summary_text(row)
        topic_l1 = "" if is_empty(row.get(COL_TOPIC_L1, "")) else str(row.get(COL_TOPIC_L1, ""))
        topic_l2 = "" if is_empty(row.get(COL_TOPIC_L2, "")) else str(row.get(COL_TOPIC_L2, ""))

        try:
            result = classify_y_once(
                client=client,
                categories=categories,
                dependent_y=original_y,
                title=title,
                theoretical_summary=summary,
                topic_l1=topic_l1,
                topic_l2=topic_l2,
                normalized_y=normalized_y,
            )

            y_to_category[normalized_y] = result["y_category"]
            success_count += 1
            print(f"[OK] {original_y} -> {result['y_category']}")

        except Exception as e:
            print(f"[ERROR] {original_y}: {e}")
            y_to_category[normalized_y] = ""

            if "insufficient_quota" in str(e) or "exceeded your current quota" in str(e):
                print("API quota exceeded. Stopping further requests.")
                break

    # 분류 결과를 전체 target row에 반영
    df["_normalized_y"] = df[COL_Y].apply(normalize_y if COL_Y in df.columns else lambda x: "")
    for idx in df[target_mask].index:
        norm_y = df.at[idx, "_normalized_y"]
        category = y_to_category.get(norm_y, "")
        if category:
            df.at[idx, COL_Y_CATEGORY] = category

    df.drop(columns=["_normalized_y"], inplace=True, errors="ignore")

    if success_count > 0:
        df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")
        print(f"Updated Excel saved to: {EXCEL_PATH}")
    else:
        print("No successful classifications. Excel was not overwritten.")


if __name__ == "__main__":
    main()