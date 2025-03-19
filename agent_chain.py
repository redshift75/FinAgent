import os
import time

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# Environment setup (replace with your actual credentials)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neoneoneo")

# Initialize LLM with shorter timeout
llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4o", 
    api_key=OPENAI_API_KEY,
    request_timeout=30  # 30 second timeout
)

# Initialize Neo4j graph
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    # Test connection
    graph.query("RETURN 1 AS test LIMIT 1")
    print("Neo4j connection successful")
except Exception as e:
    print(f"Error connecting to Neo4j: {str(e)}")
    graph = None

# Get the graph schema for GraphCypherQAChain
def get_graph_schema():
    if not graph:
        return "Error: Neo4j database is not connected."
    
    try:
        # Get node labels (types)
        node_labels = graph.query("CALL db.labels() YIELD label RETURN collect(label) as labels")[0]["labels"]
        
        # Get relationship types
        rel_types = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")[0]["types"]
        
        # Get property keys
        prop_keys = graph.query("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys")[0]["keys"]
        
        # Format the schema
        #schema = "Node Labels: " + ", ".join(node_labels) + "\n"
        #schema += "Relationship Types: " + ", ".join(rel_types) + "\n"
        #schema += "Property Keys: " + ", ".join(prop_keys) + "\n"
        
        schema += """
        The knowledge graph has the following structure:
        - Nodes with the following categories:
            - ORG: Organizations other than government or regulatory bodies (e.g., "Apple Inc.")
            - ORG_GOV: Government bodies (e.g., "United States Government")
            - ORG_REG: Regulatory bodies (e.g., "Federal Reserve")
            - PERSON: Individuals (e.g., "Elon Musk")
            - GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Germany")
            - COMP: Companies (e.g., "Google")
            - PRODUCT: Products or services (e.g., "iPhone")
            - EVENT: Specific and Material Events (e.g., "Olympic Games", "Covid-19")
            - SECTOR: Company sectors or industries (e.g., "Technology sector")
            - ECON_INDICATOR: Economic indicators (e.g., "Inflation rate")
            - FIN_INSTRUMENT: Financial and market instruments (e.g., "Stocks")
        - Relationships of the following types:
            - Relate_To
            - Operate_In
            - Impact
            - Has
            - Negative_Impact_On
            - Raise
            - Is_Member_Of
            - Participates_In
            - Announce
            - Invests_In
            - Produce
            - Introduce
            - Decrease
            - Positive_Impact_On
        - Properties: name
        """
        
        return schema
    except Exception as e:
        return f"Error exploring graph structure: {str(e)}"

# Initialize the GraphCypherQAChain
graph_qa_chain = None
if graph:
    graph_schema = get_graph_schema()
    graph_qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        schema=graph_schema,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        top_k=10
    )

# Function to invoke the GraphCypherQAChain with a user question
def ask_economic_question(question: str) -> str:
    """Ask a question to the economic agent using GraphCypherQAChain and get a response"""
    if not graph_qa_chain:
        return "Error: GraphCypherQAChain not initialized. Check Neo4j connection."
    
    print(f"Processing question: {question}")
    start_time = time.time()
    
    try:
        # Execute the chain with a timeout
        result = graph_qa_chain.invoke(
            {"query": question}, 
            {"timeout": 60}  # 60 second overall timeout
        )
        
        print(f"Query execution completed in {time.time() - start_time:.2f} seconds")
        
        # Check if we got intermediate steps and a proper output
        if isinstance(result, dict) and "result" in result:
            # If return_intermediate_steps is True, we'll get a dict with result and intermediate_steps
            response = result["result"]
            
            # Print intermediate steps for debugging
            if "intermediate_steps" in result:
                print("\nIntermediate Steps:")
                for step in result["intermediate_steps"]:
                    if isinstance(step, dict):
                        if "query" in step:
                            print(f"Cypher Query: {step['query']}")
                        if "result" in step:
                            print(f"Result: {step['result']}")
                    else:
                        print(f"Step: {step}")
        else:
            # Direct string response
            response = result
        
        return response
        
    except TimeoutError:
        return "The query timed out. Please try a simpler question or break it down into smaller parts."
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}"

# Example usage
if __name__ == "__main__":
    sample_questions = [
        "What affects the national debt?",
        "Is there any relationship between Federal Deficits and the U.S. Federal Reserve?",
        "What would happen to the economy if the Federal Reserve raises interest rates by 0.5%?",
        "Whats Jerome Powell's role?",
        "What companies are most affected by Donald Trump's policies?",
        "What is the relationship between the U.S. Federal Reserve and the U.S. government?",
        "How might a government shutdown affect commodity prices?"
        #"How are chinese steelmakers connected to Donald Trump?"
        #"Who is most responsible for the US trade deficit?"
        #"How do tariffs affect the US consumer?"
        #"What are the biggest drivers of consumer spending?"
        #"What is the relationship between U.S. immigration policies, asylum systems, and economic outcomes?"
        #"What is the connection between food shortages, immigration patterns, and economic indicators?"
        #"How do central banks' decisions influence inflation and economic growth?"
        #"Trace the impact chain from Federal Reserve decisions to employment indicators."
        #"How do sanctions against specific countries affect global economic indicators through multiple pathways?"
        #"Which government organizations have the most control over economic indicators?"
        #"Which economic concept has the most connections to other entities?"
        #"What is the complete sphere of influence of the U.S. Federal Reserve?"
        #"Identify all potential contagion pathways through which a shock in one market (e.g., housing) could propagate through financial institutions, regulatory responses, credit markets, and eventually impact multiple economic sectors."
        #"How would increased regulation of major technology companies propagate through market valuations, investment patterns, innovation metrics, and eventually impact broader economic indicators?"
        #"What are the comprehensive multi-layer impacts of NAFTA/USMCA changes on cross-border supply chains, manufacturing employment, consumer prices, and regional economic competitiveness?"
        #"What is the impact of the USMCA on the US economy?"
    ]

    for question in sample_questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {ask_economic_question(question)}")
        print("-" * 80)