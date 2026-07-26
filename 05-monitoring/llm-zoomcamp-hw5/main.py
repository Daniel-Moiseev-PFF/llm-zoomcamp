from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sqlite_helper import SQLiteSpanExporter
import starter
from rag_helper import RAGBase

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")


def calculate_cost(model, usage):
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("cost", calculate_cost(self.model, usage))
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)


traced_rag = RAGTraced(index=starter.index, llm_client=starter.client)


def main():
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = traced_rag.rag(query)
    print(answer)


if __name__ == "__main__":
    main()

#