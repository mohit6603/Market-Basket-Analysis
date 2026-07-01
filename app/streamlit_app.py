"""Market Basket Analysis — interactive rule explorer.

Loads the PRE-COMPUTED rules (data/rules.csv) once and filters them live. Nothing
is re-mined on a slider move: the expensive work (mining at min_support=0.01) already
happened offline in src/build_rules.py, so every interaction is an in-memory filter.

Run locally:  streamlit run app/streamlit_app.py   (from the project root)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

# ITEM_SEP must match src/mining.py. Chosen because no product Description contains
# "|" (68 contain commas), so splitting itemset strings on it is unambiguous.
ITEM_SEP = " | "
RULES_CSV = Path(__file__).resolve().parent.parent / "data" / "rules.csv"

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")


@st.cache_data
def load_rules(path: Path = RULES_CSV) -> pd.DataFrame:
    """Load precomputed rules and pre-split itemset strings into lists (cached once)."""
    df = pd.read_csv(path)
    df["antecedent_items"] = df["antecedents"].str.split(ITEM_SEP, regex=False)
    df["consequent_items"] = df["consequents"].str.split(ITEM_SEP, regex=False)
    df["antecedent_len"] = df["antecedent_items"].str.len()
    df["consequent_len"] = df["consequent_items"].str.len()
    return df


rules = load_rules()

# ---------------------------------------------------------------- header
st.title("🛒 Market Basket Analysis — Association Rule Explorer")
st.caption(
    "UCI *Online Retail* · 19,773 baskets · 3,986 products. "
    "This is **unsupervised association-rule mining** — there is no train/test split "
    "and **no accuracy/precision metric**. Rules are ranked by support, confidence, and lift."
)

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filter rules")
st.sidebar.caption("Rules are pre-mined at support ≥ 0.01 / confidence ≥ 0.1; sliders filter *up* from there.")

min_support = st.sidebar.slider(
    "Min support", 0.01, float(round(rules["support"].max(), 3)), 0.01, 0.005,
    help="Fraction of baskets containing the whole itemset (reach).",
)
min_confidence = st.sidebar.slider(
    "Min confidence", 0.10, 1.0, 0.30, 0.05,
    help="P(consequent | antecedent) — reliability of the rule.",
)
min_lift = st.sidebar.slider(
    "Min lift", 1.0, float(round(rules["lift"].max())), 3.0, 0.5,
    help="How many times more likely than chance. >1 positive, =1 independent, <1 substitutes.",
)

mask = (
    (rules["support"] >= min_support)
    & (rules["confidence"] >= min_confidence)
    & (rules["lift"] >= min_lift)
)
view = rules[mask].copy()

# ---------------------------------------------------------------- KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rules (filtered)", f"{len(view):,}", f"of {len(rules):,} total")
c2.metric("Max lift", f"{view['lift'].max():.1f}" if len(view) else "—")
c3.metric("Max confidence", f"{view['confidence'].max():.0%}" if len(view) else "—")
unique_items = pd.unique(rules["antecedent_items"].explode())
c4.metric("Distinct antecedent items", f"{len(unique_items):,}")

if view.empty:
    st.warning("No rules match these filters — loosen the sliders on the left.")
    st.stop()

tab_rules, tab_reco, tab_net, tab_actions = st.tabs(
    ["📋 Rules table", "🔎 Recommendations", "🕸️ Network", "💡 Business actions"]
)

# ---------------------------------------------------------------- tab 1: table + scatter
with tab_rules:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Filtered rules")
        st.dataframe(
            view[["antecedents", "consequents", "support", "confidence", "lift", "leverage"]]
            .sort_values("lift", ascending=False)
            .style.format({"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.1f}", "leverage": "{:.4f}"}),
            width="stretch",
            height=460,
        )
    with right:
        st.subheader("Support vs confidence (colour = lift)")
        fig, ax = plt.subplots(figsize=(5, 4.6))
        sc = ax.scatter(
            view["support"], view["confidence"],
            c=view["lift"], s=view["lift"] * 5, cmap="viridis", alpha=0.6, edgecolors="none",
        )
        ax.set_xlabel("support")
        ax.set_ylabel("confidence")
        fig.colorbar(sc, ax=ax, label="lift")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Top-left = rare but reliable; brighter/larger = stronger lift.")

# ---------------------------------------------------------------- tab 2: recommendations
with tab_reco:
    st.subheader("Customers who bought X also bought…")
    st.caption("Ranked by **confidence** (reliability), not lift — confidence tells you which way to point a recommendation.")
    # dropdown = items that appear as a single-item antecedent (uses full pool, not the sliders)
    single = rules[rules["antecedent_len"] == 1].copy()
    single["item"] = single["antecedent_items"].str[0]
    items = sorted(single["item"].unique())
    choice = st.selectbox("Pick a product", items, index=items.index("HERB MARKER THYME") if "HERB MARKER THYME" in items else 0)

    recs = (
        single[single["item"] == choice]
        .sort_values("confidence", ascending=False)
        [["consequents", "confidence", "lift", "support"]]
        .head(10)
        .rename(columns={"consequents": "also bought"})
    )
    if recs.empty:
        st.info("No recommendations for this item in the pool.")
    else:
        st.dataframe(
            recs.style.format({"confidence": "{:.0%}", "lift": "{:.1f}", "support": "{:.3f}"}),
            width="stretch", hide_index=True,
        )

# ---------------------------------------------------------------- tab 3: network graph
with tab_net:
    st.subheader("Rule network (top single-item rules by lift)")
    max_edges = st.slider("Number of rules to draw", 10, 60, 30, 5)
    simple = view[(view["antecedent_len"] == 1) & (view["consequent_len"] == 1)]
    top = simple.sort_values("lift", ascending=False).head(max_edges)

    if top.empty:
        st.info("No single-item→single-item rules under the current filters — loosen them.")
    else:
        g = nx.DiGraph()
        for _, r in top.iterrows():
            g.add_edge(r["antecedents"], r["consequents"], lift=r["lift"])
        fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(g, k=0.6, seed=42)
        nx.draw_networkx_nodes(g, pos, node_size=350, node_color="#4C9BE8", alpha=0.9, ax=ax)
        nx.draw_networkx_edges(
            g, pos, ax=ax, alpha=0.4, arrows=True, arrowsize=8,
            width=[0.4 + g[u][v]["lift"] / 20 for u, v in g.edges()],
            edge_color="#888",
        )
        nx.draw_networkx_labels(g, pos, labels={n: n[:22] for n in g}, font_size=6.5, ax=ax)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Clusters = product ranges bought together (e.g. herb markers, teacup colours, jumbo bags).")

# ---------------------------------------------------------------- tab 4: business actions
with tab_actions:
    st.subheader("Top rules as business actions")
    st.markdown(
        "Ranked by **leverage** (frequency × strength = biggest absolute impact). "
        "These are **associations, not causation** — each is a hypothesis to A/B test."
    )
    top_lev = rules.sort_values("leverage", ascending=False).head(8)
    st.dataframe(
        top_lev[["antecedents", "consequents", "support", "confidence", "lift", "leverage"]]
        .style.format({"support": "{:.3f}", "confidence": "{:.0%}", "lift": "{:.1f}", "leverage": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        """
**Recommended actions**
1. **"Complete the collection" bundles** — matched sets (herb markers, Regency teacups) have very high lift; sell them as bundles with "complete the set" prompts.
2. **Directional cross-sell** — drive the recommender by *confidence*: e.g. show RED RETROSPOT to PINK POLKADOT buyers (68%), not the weaker reverse (39%).
3. **Catalog/shelf adjacency** — co-locate high-leverage pairs (teacup colours, jumbo bags).
4. **Coordinated inventory** — for high-lift sets, a stockout of one variant strands demand for the rest.
        """
    )
