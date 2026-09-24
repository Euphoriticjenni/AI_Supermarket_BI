"""Offline AI business assistant for the AI_Supermarket_BI project.

A text-to-SQL assistant that works WITHOUT any API key: business questions
are matched against a library of regex query templates, executed against
the SQLite database, and answered with string-template natural language
plus an optional plotly chart.

No ML libraries and no LLM calls — pure SQL + pandas + string formatting,
so it runs fully offline (the original spec's OpenAI path is intentionally
not required).
"""
import re
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

PALETTE = px.colors.qualitative.Set2

#: Column-name fragments that mark a result column as temporal (line charts).
_DATE_HINTS = ("date", "month", "hour", "week", "year", "quarter", "day")


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------
class BusinessAssistant:
    """Rule-based text-to-SQL business Q&A over the supermarket database.

    Usage::

        assistant = BusinessAssistant()
        result = assistant.ask("What is the total revenue?")
        print(result["answer"])   # natural language
        result["data"]            # DataFrame
        result["sql"]             # the executed query
        result["chart"]           # plotly figure or None
    """

    def __init__(self, db_path: Optional[str] = None):
        """Create the engine and build the query-template library.

        Parameters
        ----------
        db_path:
            Path to the SQLite file; defaults to ``config.DATABASE_PATH``.
        """
        self.db_path = str(db_path or config.DATABASE_PATH)
        self.engine = create_engine(f"sqlite:///{self.db_path.replace(chr(92), '/')}")
        self.templates: Dict[str, str] = self._build_query_templates()
        logger.info(
            "BusinessAssistant initialised on {} with {} query templates.",
            self.db_path, len(self.templates),
        )

    # ------------------------------------------------------------------
    # Template library
    # ------------------------------------------------------------------
    def _build_query_templates(self) -> Dict[str, str]:
        """Return the regex-pattern -> SQL-query library (ordered).

        Order matters: patterns are tried top to bottom and the FIRST match
        wins, so specific multi-word patterns are placed before generic
        single-word ones (e.g. "membership" before "member", "stock" before
        "how many").
        """
        return {
            # --- specific customer / product phrasings first -------------
            r"top customers|best customers|vip":
                "SELECT Customer_ID, ROUND(Total_Spent,2) as Total_Spent, "
                "Visit_Count, Membership FROM customers "
                "ORDER BY Total_Spent DESC LIMIT 20",

            r"top products|best selling|best product|top product":
                "SELECT Product_Line, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders, ROUND(AVG(Total),2) as Avg_Order "
                "FROM sales GROUP BY Product_Line ORDER BY Revenue DESC LIMIT 10",

            r"worst|slow|underperform|bottom":
                "SELECT Product_Line, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Product_Line ORDER BY Revenue ASC LIMIT 5",

            r"stock|inventory|reorder|restock":
                "SELECT Product_ID, Product_Name, Stock, Reorder_Level, "
                "CASE WHEN Stock = 0 THEN 'STOCKOUT' "
                "WHEN Stock < Reorder_Level THEN 'LOW' ELSE 'OK' END as Status "
                "FROM products ORDER BY Stock ASC LIMIT 20",

            r"membership":
                "SELECT Membership, COUNT(*) as Count, "
                "ROUND(AVG(Total_Spent),2) as Avg_Spent "
                "FROM customers GROUP BY Membership ORDER BY Count DESC",

            r"customer type|member|customer_type":
                "SELECT Customer_Type, COUNT(DISTINCT Invoice_ID) as Orders, "
                "ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Customer_Type",

            r"demographics|age|customer info":
                "SELECT Gender, ROUND(AVG(Age),1) as Avg_Age, COUNT(*) as Count "
                "FROM customers GROUP BY Gender",

            r"gender":
                "SELECT Gender, COUNT(DISTINCT Invoice_ID) as Orders, "
                "ROUND(SUM(Total),2) as Revenue FROM sales GROUP BY Gender",

            r"average order|avg order|basket|aov":
                "SELECT ROUND(AVG(Total),2) as Avg_Order_Value, "
                "ROUND(MIN(Total),2) as Min_Order, ROUND(MAX(Total),2) as Max_Order, "
                "ROUND(SUM(Total)/COUNT(DISTINCT Invoice_ID),2) as Revenue_Per_Order "
                "FROM sales",

            r"payment|payment method":
                "SELECT Payment, COUNT(*) as Transactions, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Payment ORDER BY Transactions DESC",

            r"rating|satisfaction|review":
                "SELECT Branch, ROUND(AVG(Rating),2) as Avg_Rating, COUNT(*) as Reviews "
                "FROM sales GROUP BY Branch ORDER BY Avg_Rating DESC",

            r"hourly|hour|time of day":
                "SELECT strftime('%H', Time) as Hour, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Hour ORDER BY Hour",

            r"weekday|day of week|weekend":
                "SELECT CASE WHEN strftime('%w', Date) IN ('0','6') "
                "THEN 'Weekend' ELSE 'Weekday' END as Day_Type, "
                "ROUND(SUM(Total),2) as Revenue, COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY Day_Type",

            # NOTE: placed after weekday/weekend so "weekday sales" is not
            # captured by the "day sales" alternative below.
            r"daily|day trend|day sales":
                "SELECT Date, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Date ORDER BY Date",

            r"monthly|month trend|month sales":
                "SELECT strftime('%Y-%m', Date) as Month, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY Month ORDER BY Month",

            r"seasonal|season|quarter|quarterly":
                "SELECT strftime('%Y', Date) as Year, "
                "CASE WHEN CAST(strftime('%m',Date) AS INT) BETWEEN 1 AND 3 THEN 'Q1' "
                "WHEN CAST(strftime('%m',Date) AS INT) BETWEEN 4 AND 6 THEN 'Q2' "
                "WHEN CAST(strftime('%m',Date) AS INT) BETWEEN 7 AND 9 THEN 'Q3' "
                "ELSE 'Q4' END as Quarter, "
                "ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Year, Quarter ORDER BY Year, Quarter",

            r"yoy|year over year|year comparison|yearly|compare year|by year":
                "SELECT strftime('%Y', Date) as Year, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY Year ORDER BY Year",

            r"branch|store|branches":
                "SELECT Branch, City, ROUND(SUM(Total),2) as Revenue, "
                "ROUND(SUM(Gross_Income),2) as Profit, "
                "COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY Branch, City ORDER BY Revenue DESC",

            r"city|cities":
                "SELECT City, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY City ORDER BY Revenue DESC",

            r"profit|gross income|total profit":
                "SELECT ROUND(SUM(Gross_Income),2) as Total_Profit FROM sales",

            r"revenue|total sales|total revenue":
                "SELECT ROUND(SUM(Total),2) as Total_Revenue FROM sales",

            r"tax":
                "SELECT ROUND(SUM(Tax_5pct),2) as Total_Tax, "
                "ROUND(SUM(COGS),2) as Total_COGS FROM sales",

            r"margin":
                "SELECT ROUND(AVG(Gross_Margin_Pct),2) as Avg_Margin_Pct FROM sales",

            r"peak|best hour|busiest":
                "SELECT strftime('%H', Time) as Hour, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders FROM sales "
                "GROUP BY Hour ORDER BY Revenue DESC LIMIT 5",

            r"product line|category|categories":
                "SELECT Product_Line, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Invoice_ID) as Orders "
                "FROM sales GROUP BY Product_Line ORDER BY Revenue DESC",

            # generic "how many" deliberately LAST among content patterns.
            r"count|how many|total orders|order count":
                "SELECT COUNT(DISTINCT Invoice_ID) as Total_Orders, "
                "COUNT(*) as Total_Transactions FROM sales",

            r"overview|summary|kpi|dashboard":
                "SELECT COUNT(DISTINCT Invoice_ID) as Total_Orders, "
                "ROUND(SUM(Total),2) as Total_Revenue, "
                "ROUND(SUM(Gross_Income),2) as Total_Profit, "
                "ROUND(AVG(Total),2) as Avg_Order_Value, "
                "ROUND(AVG(Rating),2) as Avg_Rating FROM sales",
        }

    # ------------------------------------------------------------------
    # Matching & execution
    # ------------------------------------------------------------------
    def _match_template(self, question: str) -> Tuple[str, str]:
        """Find the first template whose regex matches the question.

        Returns
        -------
        tuple(str, str)
            ``(sql, pattern)`` — the query to run and the pattern that
            matched (``"default_overview"`` when nothing matched).
        """
        question_lower = question.lower()
        for pattern, sql in self.templates.items():
            if re.search(pattern, question_lower):
                return sql, pattern
        default_sql = self.templates[r"overview|summary|kpi|dashboard"]
        return default_sql, "default_overview"

    def _run_query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL string against the engine and return a DataFrame."""
        return pd.read_sql(sql, self.engine)

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_value(value, column: str = "") -> str:
        """Format a cell for natural-language output (money-aware).

        Integral floats under year/hour/count-like columns print as plain
        integers (so years show as "2023", not "2,023.00"); other numbers
        keep thousands separators and two decimals.
        """
        column_lower = str(column).lower()
        plain_int = any(
            hint in column_lower
            for hint in ("year", "hour", "month", "week", "quarter", "date", "day")
        )
        if isinstance(value, float):
            if value.is_integer() and plain_int:
                return str(int(value))
            return f"{value:,.2f}"
        try:
            return str(int(value)) if plain_int else f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def _format_answer(self, df: pd.DataFrame) -> str:
        """Turn a query result into a natural-language answer by shape."""
        if df.empty:
            return "No data found for that question."

        if df.shape == (1, 1):
            column, value = df.columns[0], df.iloc[0, 0]
            money = any(k in column.lower() for k in ("revenue", "profit", "total", "spent", "tax", "order"))
            if isinstance(value, (int, float)) and money:
                return f"The {column.replace('_', ' ').lower()} is ${self._fmt_value(value, column)}."
            return f"The {column.replace('_', ' ').lower()} is {self._fmt_value(value, column)}."

        if len(df) == 1:
            parts = [f"{col.replace('_', ' ')} = {self._fmt_value(val, col)}"
                     for col, val in df.iloc[0].items()]
            return "Here's the summary: " + ", ".join(parts) + "."

        # Multiple rows: top-3 preview plus a max/min comparison.
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        lines = []
        for _, row in df.head(3).iterrows():
            lines.append(" - " + ", ".join(
                f"{col.replace('_', ' ')}: {self._fmt_value(val, col)}" for col, val in row.items()
            ))
        answer = f"Here are the top {min(3, len(df))} of {len(df)} results:\n" + "\n".join(lines)

        if numeric_cols:
            metric = numeric_cols[0]
            label_col = df.columns[0]
            best_idx = df[metric].idxmax()
            worst_idx = df[metric].idxmin()
            best_label = str(df.loc[best_idx, label_col])
            worst_label = str(df.loc[worst_idx, label_col])
            if best_label != worst_label:
                diff = df[metric].max() - df[metric].min()
                answer += (
                    f"\n{best_label} leads with {self._fmt_value(df[metric].max(), metric)} "
                    f"({self._fmt_value(diff, metric)} more than {worst_label})."
                )
        return answer

    def _chart_from_result(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Best-effort plotly chart for a query result (None on failure).

        Heuristics: a temporal + numeric column pair becomes a line chart;
        a categorical + numeric pair becomes a pie (2-4 categories) or a
        bar chart.
        """
        try:
            if df.empty or df.shape[1] < 2:
                return None
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                return None
            value_col = numeric_cols[0]
            label_col = df.columns[0]

            label_lower = str(label_col).lower()
            temporal = any(hint in label_lower for hint in _DATE_HINTS)

            chart_df = df.copy()
            if temporal:
                chart_df[label_col] = chart_df[label_col].astype(str)
                chart_df = chart_df.sort_values(label_col)
                fig = px.line(
                    chart_df, x=label_col, y=value_col, markers=True,
                    color_discrete_sequence=PALETTE[:1],
                    title=f"{value_col.replace('_', ' ')} by {label_col.replace('_', ' ')}",
                )
            else:
                n_categories = chart_df[label_col].nunique()
                if 2 <= n_categories <= 4:
                    fig = px.pie(
                        chart_df, names=label_col, values=value_col,
                        color_discrete_sequence=PALETTE, hole=0.4,
                        title=f"{value_col.replace('_', ' ')} by {label_col.replace('_', ' ')}",
                    )
                else:
                    fig = px.bar(
                        chart_df, x=label_col, y=value_col,
                        color=label_col, color_discrete_sequence=PALETTE,
                        title=f"{value_col.replace('_', ' ')} by {label_col.replace('_', ' ')}",
                    )
                    fig.update_layout(showlegend=False)
            fig.update_layout(template=config.PLOTLY_TEMPLATE, height=420)
            return fig
        except Exception as exc:
            logger.debug("Chart generation skipped: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str) -> Dict:
        """Answer a natural-language business question.

        Returns
        -------
        dict
            ``{"answer": str, "sql": str, "data": DataFrame,
            "chart": plotly Figure or None}``.
        """
        logger.info("Assistant question: '{}'", question)
        try:
            sql, pattern = self._match_template(question)
            data = self._run_query(sql)
            answer = self._format_answer(data)
            chart = self._chart_from_result(data)
            logger.info("Matched pattern '{}' -> {} rows.", pattern, len(data))
            return {"answer": answer, "sql": sql, "data": data, "chart": chart}
        except Exception as exc:
            logger.exception("Assistant failed on question: '{}'", question)
            return {
                "answer": f"Sorry, I couldn't answer that ({type(exc).__name__}: {exc}). "
                          "Try rephrasing, e.g. 'What is the total revenue?'",
                "sql": "",
                "data": pd.DataFrame(),
                "chart": None,
            }

    def get_executive_summary(self) -> str:
        """Compile a multi-paragraph executive summary with concrete numbers."""
        try:
            overview = self._run_query(
                "SELECT COUNT(DISTINCT Invoice_ID) as Total_Orders, "
                "ROUND(SUM(Total),2) as Total_Revenue, "
                "ROUND(SUM(Gross_Income),2) as Total_Profit, "
                "ROUND(AVG(Total),2) as Avg_Order_Value, "
                "ROUND(AVG(Rating),2) as Avg_Rating FROM sales"
            ).iloc[0]
            branches = self._run_query(
                "SELECT Branch, City, ROUND(SUM(Total),2) as Revenue FROM sales "
                "GROUP BY Branch, City ORDER BY Revenue DESC"
            )
            products = self._run_query(
                "SELECT Product_Line, ROUND(SUM(Total),2) as Revenue FROM sales "
                "GROUP BY Product_Line ORDER BY Revenue DESC"
            )
            customers = self._run_query(
                "SELECT Customer_Type, COUNT(DISTINCT Invoice_ID) as Orders, "
                "ROUND(SUM(Total),2) as Revenue FROM sales GROUP BY Customer_Type"
            )
            years = self._run_query(
                "SELECT strftime('%Y', Date) as Year, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Year ORDER BY Year"
            )

            top_branch = branches.iloc[0]
            top_product = products.iloc[0]
            member = customers[customers["Customer_Type"] == "Member"].iloc[0]
            normal = customers[customers["Customer_Type"] == "Normal"].iloc[0]
            growth = (
                (years.iloc[-1]["Revenue"] / years.iloc[0]["Revenue"] - 1) * 100
                if len(years) > 1 else 0.0
            )

            summary = (
                f"EXECUTIVE SUMMARY\n"
                f"=================\n"
                f"Across the full period the chain processed "
                f"{int(overview['Total_Orders']):,} orders for total revenue of "
                f"${overview['Total_Revenue']:,.2f}, earning gross profit of "
                f"${overview['Total_Profit']:,.2f}. The average order value was "
                f"${overview['Avg_Order_Value']:,.2f} and the average customer "
                f"rating was {overview['Avg_Rating']:.2f}/10.\n\n"
                f"BRANCH PERFORMANCE\n"
                f"Branch {top_branch['Branch']} ({top_branch['City']}) is the top "
                f"revenue generator with ${top_branch['Revenue']:,.2f}. "
                f"The strongest product line is {top_product['Product_Line']} at "
                f"${top_product['Revenue']:,.2f} in revenue.\n\n"
                f"CUSTOMERS\n"
                f"Members placed {int(member['Orders']):,} orders "
                f"(${member['Revenue']:,.2f}) versus {int(normal['Orders']):,} orders "
                f"(${normal['Revenue']:,.2f}) for non-members — members generate "
                f"{(member['Revenue'] / normal['Revenue'] - 1) * 100:.1f}% more revenue.\n\n"
                f"GROWTH\n"
                f"Revenue moved from ${years.iloc[0]['Revenue']:,.2f} in "
                f"{years.iloc[0]['Year']} to ${years.iloc[-1]['Revenue']:,.2f} in "
                f"{years.iloc[-1]['Year']}, a change of {growth:+.1f}%."
            )
            logger.info("Executive summary generated ({} chars).", len(summary))
            return summary
        except Exception:
            logger.exception("Executive summary failed")
            return "Executive summary is unavailable — please check the database."

    def get_insights(self) -> list:
        """Generate 5-7 concrete business insights from analytical queries."""
        insights = []
        try:
            # 1. Year-over-year revenue growth.
            years = self._run_query(
                "SELECT strftime('%Y', Date) as Year, ROUND(SUM(Total),2) as Revenue "
                "FROM sales GROUP BY Year ORDER BY Year"
            )
            if len(years) > 1:
                pct = (years.iloc[-1]["Revenue"] / years.iloc[0]["Revenue"] - 1) * 100
                insights.append(
                    f"Revenue {'grew' if pct >= 0 else 'declined'} {abs(pct):.1f}% "
                    f"from {years.iloc[0]['Year']} to {years.iloc[-1]['Year']}."
                )

            # 2. Branch spread.
            branches = self._run_query(
                "SELECT Branch, ROUND(SUM(Total),2) as Revenue FROM sales "
                "GROUP BY Branch ORDER BY Revenue"
            )
            if len(branches) > 1:
                pct = (branches.iloc[-1]["Revenue"] / branches.iloc[0]["Revenue"] - 1) * 100
                insights.append(
                    f"Branch {branches.iloc[-1]['Branch']} outperforms "
                    f"Branch {branches.iloc[0]['Branch']} by {pct:.1f}% in revenue."
                )

            # 3. Top product line share.
            totals = self._run_query(
                "SELECT Product_Line, ROUND(SUM(Total),2) as Revenue FROM sales "
                "GROUP BY Product_Line ORDER BY Revenue DESC"
            )
            share = totals.iloc[0]["Revenue"] / totals["Revenue"].sum() * 100
            insights.append(
                f"{totals.iloc[0]['Product_Line']} leads with {share:.1f}% of total revenue."
            )

            # 4. Weekend vs weekday.
            days = self._run_query(
                "SELECT CASE WHEN strftime('%w', Date) IN ('0','6') THEN 'Weekend' "
                "ELSE 'Weekday' END as Day_Type, ROUND(SUM(Total),2) as Revenue, "
                "COUNT(DISTINCT Date) as nDays FROM sales GROUP BY Day_Type"
            ).set_index("Day_Type")
            if {"Weekend", "Weekday"} <= set(days.index):
                weekend_daily = days.loc["Weekend", "Revenue"] / days.loc["Weekend", "nDays"]
                weekday_daily = days.loc["Weekday", "Revenue"] / days.loc["Weekday", "nDays"]
                pct = (weekend_daily / weekday_daily - 1) * 100
                insights.append(
                    f"Weekend daily sales are {pct:.1f}% higher than weekday daily sales."
                )

            # 5. Average rating.
            rating = self._run_query("SELECT ROUND(AVG(Rating),2) as r FROM sales").iloc[0]["r"]
            insights.append(f"Average customer rating is {rating:.2f}/10.")

            # 6. Stockouts.
            stockouts = self._run_query(
                "SELECT COUNT(*) as n FROM products WHERE Stock = 0"
            ).iloc[0]["n"]
            insights.append(f"{int(stockouts)} products are currently out of stock.")

            # 7. Member vs normal revenue.
            cust = self._run_query(
                "SELECT Customer_Type, ROUND(SUM(Total),2) as Revenue FROM sales "
                "GROUP BY Customer_Type"
            ).set_index("Customer_Type")
            if {"Member", "Normal"} <= set(cust.index):
                pct = (cust.loc["Member", "Revenue"] / cust.loc["Normal", "Revenue"] - 1) * 100
                insights.append(
                    f"Members generate {pct:.1f}% more revenue than non-members."
                )
        except Exception:
            logger.exception("Insight generation partially failed")
        if not insights:
            insights.append("No insights could be generated — check the database.")
        logger.info("Generated {} insights.", len(insights))
        return insights


if __name__ == "__main__":
    assistant = BusinessAssistant()
    print(assistant.ask("What is the total revenue?")["answer"])
    print(assistant.ask("Show me the top products")["answer"])
    print(assistant.get_insights())
