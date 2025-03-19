import os
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from pydantic import BaseModel, Field
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_core.tools import BaseTool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

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

# Query cache to store previous results
query_cache = {}
# Track queries already attempted in a session
tried_queries = set()

# State definition for the agent
class AgentState(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    query_plan: Optional[str] = None
    knowledge_results: List[Dict] = Field(default_factory=list)
    next_action: Optional[str] = None
    intermediate_steps: List[Tuple[str, str]] = Field(default_factory=list)
    tried_queries: Set[str] = Field(default_factory=set)

# Define schema for KnowledgeGraphQueryTool
class KnowledgeGraphQuerySchema(BaseModel):
    query: str = Field(..., description="The Cypher query to execute against the Neo4j knowledge graph")

# Define schema for ExploreGraphStructureTool
class ExploreGraphStructureSchema(BaseModel):
    dummy: str = Field(default="", description="This parameter is not used but is required for the tool to work properly")

# Define schema for GenerateResponseTool
class GenerateResponseSchema(BaseModel):
    content: str = Field(..., description="The content to transform into a final response")

# Custom tools for the agent
class KnowledgeGraphQueryTool(BaseTool):
    name: str = "query_knowledge_graph"
    description: str = """Use this tool to query the Neo4j knowledge graph about economic concepts, relationships, and data.
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
        
    Input should be a Cypher query that will extract the most relevant information to answer the user's question.

    Some helpful Cypher query patterns:
    1. Find path relationships between two entities:
       MATCH (a)-[p*1..2]-(b) WHERE toLower(a.name) CONTAINS toLower('Entity1') AND toLower(b.name) CONTAINS toLower('Entity2') RETURN a, p, b LIMIT 5
    
    2. Find all entities related to a concept:
       MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS toLower('Concept') RETURN a, r, b LIMIT 10
    
    3. Find entities with a specific relationship:
       MATCH (a)-[r:RELATIONSHIP_TYPE]-(b) RETURN a, r, b LIMIT 10

    IMPORTANT: ALWAYS use toLower() for case-insensitive matching in CONTAINS clauses, like:
       WHERE toLower(a.name) CONTAINS toLower('searchTerm')
       
    IMPORTANT: If a query does not provide a direct relationship between the entities, use path queries to find a path between the entities.
    IMPORTANT: Always add a LIMIT clause (e.g., LIMIT 10) to your queries to avoid returning too many results.
    For path queries, use a maximum path length of 2 (e.g., [*1..2]) to avoid performance issues.
         
    IMPORTANT: DO NOT REPEAT QUERIES that have already been tried. If a query returns no results, try a different query approach instead of repeating similar patterns.
    """
    
    args_schema: type[BaseModel] = KnowledgeGraphQuerySchema
    
    def _run(self, query: str) -> str:
        global tried_queries
        
        if not graph:
            return "Error: Neo4j database is not connected."
        
        # Check if this query or a very similar one has already been tried
        cleaned_query = ' '.join(query.lower().split())
        if cleaned_query in tried_queries:
            return "This query is too similar to one already tried. Please try a different approach instead of repeating similar queries."
        
        # Add to tried queries
        tried_queries.add(cleaned_query)
            
        # Check if query is in cache
        if query in query_cache:
            return query_cache[query]
        
        # Add a safety LIMIT if none exists
        if "LIMIT" not in query.upper():
            query += " LIMIT 10"
        
        try:
            # Set a timeout for the query execution
            start_time = time.time()
            timeout = 15  # 15 second timeout
            
            # Use the Neo4jGraph's query method directly
            results = graph.query(query)
            
            # Check if timeout exceeded
            if time.time() - start_time > timeout:
                return "Query execution took too long. Please simplify your query."
            
            # Format results
            if not results:
                response = "No results found for this query. Please try a different query with different search terms or relationships."
            else:
                # Limit the size of the response to avoid overwhelming the LLM
                if len(json.dumps(results)) > 8000:
                    # Truncate results if too large
                    results = results[:5] if isinstance(results, list) else results
                    response = json.dumps(results, indent=2) + "\n\n(Results truncated due to size. Please refine your query.)"
                else:
                    response = json.dumps(results, indent=2)
            
            # Cache the result
            query_cache[query] = response
            return response
        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            
            # Provide guidance on how to fix common Cypher query errors
            if "SyntaxError" in str(e):
                error_message += "\n\nSyntax error in Cypher query. Check for missing parentheses, brackets, or incorrect keywords."
            elif "not found" in str(e).lower():
                error_message += "\n\nProperty or relationship type not found. Make sure you're using only the relationships and properties defined in the knowledge graph."
            
            return error_message
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

class ExploreGraphStructureTool(BaseTool):
    name: str = "explore_graph_structure"
    description: str = """Use this tool to explore the structure of the knowledge graph.
    This will help you understand the types of nodes, relationships, and properties available.
    No parameters are needed - just call the tool to get information about the graph structure.
    """
    
    args_schema: type[BaseModel] = ExploreGraphStructureSchema
    
    def _run(self, dummy: str = "") -> str:
        if not graph:
            return "Error: Neo4j database is not connected."
            
        try:
            # Get node labels (types)
            node_labels = graph.query("CALL db.labels() YIELD label RETURN collect(label) as labels")[0]["labels"]
            
            # Get relationship types
            rel_types = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types")[0]["types"]
            
            # Get property keys
            prop_keys = graph.query("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys")[0]["keys"]
            
            # Get sample nodes for each label (limited to 3 each)
            sample_nodes = {}
            for label in node_labels:
                samples = graph.query(f"MATCH (n:{label}) RETURN n LIMIT 3")
                if samples:
                    sample_nodes[label] = [dict(node["n"]) for node in samples]
            
            # Format the results
            result = {
                "node_labels": node_labels,
                "relationship_types": rel_types,
                "property_keys": prop_keys,
                "sample_nodes": sample_nodes
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error exploring graph structure: {str(e)}"
    
    def _arun(self, dummy: str = "") -> str:
        raise NotImplementedError("Async not implemented")

# Create tools list
tools = [
    KnowledgeGraphQueryTool(),
    ExploreGraphStructureTool(),
    Tool(
        name="generate_response",
        func=lambda content: content,
        description="Use this tool to generate a final response based on the knowledge graph results",
        args_schema=GenerateResponseSchema
    )
]

# Create the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert economist tasked with answering questions based on information from an economic knowledge graph.
    
    Your goal is to:
    1. Analyze the user's question
    2. Understand the entities and relationships in the graph using the explore_graph_structure tool
    3. Create appropriate Cypher queries to extract relevant information
    4. Synthesize the results into a clear, informative response
    5. If the user's question is not related to the graph, respond with "I'm sorry, I don't have information on that topic."
    
    Guidelines for creating Cypher queries:
    - Start with simple queries and gradually refine them
    - ALWAYS add a LIMIT clause (e.g., LIMIT 10 or LIMIT 20) to your queries
    - Keep path queries short ([*1..2] maximum)
    - Use MATCH patterns like (a)-[r]-(b) to find relationships
    - Use WHERE clauses with CONTAINS for flexible string matching
    - Limit your results (e.g., LIMIT 20) for initial exploration
    - Use CASE-INSENSITIVE string matching
    - NEVER repeat the same queries that have already been tried and didn't yield results
    - If a search term doesn't work, try synonyms or related concepts
    
    Guidelines for generating responses:
    - Base your answers strictly on the knowledge graph results
    - Be transparent about relationships and causal mechanisms
    - Acknowledge limitations when data is incomplete
    - Explain economic concepts clearly while maintaining accuracy
    - Clearly indicate cause-effect relationships
    - Include relevant statistics when available
    - If multiple query attempts yield no results, use the generate_response tool to create a final response stating that information is not available
    
    Use the query_knowledge_graph tool to get information from the knowledge graph.
    Use the generate_response tool to create your final answer."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

# Create the agent executor with timeout
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    early_stopping_method="force",  # Force stop at max_iterations
    timeout=60  # 60 second overall timeout
)

# Function to invoke the agent with a user question
def ask_economic_question(question: str) -> str:
    """Ask a question to the economic agent and get a response"""
    # Clear the query cache and tried queries for a new question
    global query_cache, tried_queries
    query_cache.clear()
    tried_queries.clear()
    
    print(f"Processing question: {question}")
    start_time = time.time()
    
    try:
        # Execute the agent with just the input message
        result = agent_executor.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        print(f"Agent execution completed in {time.time() - start_time:.2f} seconds")
        
        # Check if we got a proper output
        if "output" in result and result["output"]:
            return result["output"]
        else:
            # If we hit the iteration limit but have some intermediate steps
            if "intermediate_steps" in result and result["intermediate_steps"]:
                # Try to synthesize a response from what we have
                steps = result["intermediate_steps"]
                last_results = [s[1] for s in steps if "query_knowledge_graph" in s[0] or "explore_graph_structure" in s[0]]
                
                if last_results:
                    print("Synthesizing response from intermediate steps")
                    # Use the model to synthesize what we have
                    synthesis_prompt = f"""Based on the following information from the economic knowledge graph, 
                    answer the question: "{question}"
                    
                    Information:
                    {' '.join(last_results[-3:])}
                    
                    Please provide a concise answer based only on the information above.
                    If the information is insufficient, indicate that the knowledge graph doesn't contain enough relevant data.
                    """
                    synthesis_response = llm.invoke(synthesis_prompt)
                    return synthesis_response.content
            
            return "I couldn't find enough information in the knowledge graph to answer your question. Please try rephrasing or asking a more specific question about economic relationships."
    except TimeoutError:
        return "The query timed out. Please try a simpler question or break it down into smaller parts."
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return f"Error processing question: {str(e)}"

# Example usage
if __name__ == "__main__":
    sample_questions = [
        #"What affects the national debt?"
        #"Is there any relationship between Federal Deficits and the U.S. Federal Reserve?"
        #"What would happen to the economy if the Federal Reserve raises interest rates by 0.5%?"
        #"Whats Jerome Powell's role?"
        #"What companies are most affected by Donald Trump's policies?",
        #"What is the relationship between the U.S. Federal Reserve and the U.S. government?"
        #"How might a government shutdown affect commodity prices?"
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
        "How would increased regulation of major technology companies propagate through market valuations, investment patterns, innovation metrics, and eventually impact broader economic indicators?"
        #"What are the comprehensive multi-layer impacts of NAFTA/USMCA changes on cross-border supply chains, manufacturing employment, consumer prices, and regional economic competitiveness?"
        #"What is the impact of the USMCA on the US economy?"
    ]


    
    for question in sample_questions:
        print(f"\nQuestion: {question}")
        print(f"Answer: {ask_economic_question(question)}")
        print("-" * 80)