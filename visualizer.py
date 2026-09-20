import plotly.graph_objects as go
import plotly.express as px

class Visualizer:

    def generate_all_charts(self, results, contribution):
        k_vals = [r["K"] for r in results]
        recall = [r["recall_at_k"] for r in results]
        grounding = [r["grounding_score"] for r in results]
        hallucination = [r["hallucination_rate"] for r in results]
        quality = [r["quality_score"] for r in results]

        charts = {}

        # Retrieval Metrics
        charts["retrieval_metrics"] = px.line(
            x=k_vals, y=recall, markers=True,
            labels={"x": "K", "y": "Recall"},
            title="Recall vs K"
        )

        # Response Quality
        charts["response_quality"] = px.line(
            x=k_vals, y=grounding, markers=True,
            labels={"x": "K", "y": "Grounding Score"},
            title="Grounding Score vs K"
        )

        # Recall vs Hallucination
        charts["recall_vs_hallucination"] = px.scatter(
            x=recall, y=hallucination,
            labels={"x": "Recall", "y": "Hallucination"},
            title="Recall vs Hallucination"
        )

        # Grounding vs Hallucination
        charts["grounding_vs_hallucination"] = px.scatter(
            x=grounding, y=hallucination,
            labels={"x": "Grounding", "y": "Hallucination"},
            title="Grounding vs Hallucination"
        )

        # Quality Score
        charts["quality_score"] = px.line(
            x=k_vals, y=quality, markers=True,
            labels={"x": "K", "y": "Quality Score"},
            title="Quality Score vs K"
        )

        # Contribution Chart
        if contribution:
            docs = list(contribution.keys())
            values = list(contribution.values())

            charts["doc_contribution"] = px.pie(
                names=docs, values=values,
                title="Document Contribution"
            )
        else:
            charts["doc_contribution"] = go.Figure()

        return charts

    def plot_doc_contribution(self, contribution, title="Contribution"):
        docs = list(contribution.keys())
        values = list(contribution.values())

        return px.pie(
            names=docs,
            values=values,
            title=title
        )