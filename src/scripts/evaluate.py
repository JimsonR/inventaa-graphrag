import sys
import os
import time

# Add src to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.services.agent import ask_agent, initialize_agent

# Define Ground Truth tests
# Each test consists of:
# 'name': A description of the test
# 'query': The user query to send to the agent
# 'type': Expected return type ('list' or 'str')
# 'validate': A lambda function receiving the result that returns a boolean indicating pass/fail
# 'error_msg': Message to print if validation fails
GROUND_TRUTHS = [
    {
        "name": "Structured Product Query (Direct bypass)",
        "query": "show lowest rated 2 ip65 rated products",
        "type": list,
        "validate": lambda res: len(res) > 0 and isinstance(res, list),
        "error_msg": "Expected a list containing product results."
    },
    {
        "name": "Product Details Query (Conversational)",
        "query": "What is the warranty of the Oxana Tilt COB Down LED Light?",
        "type": str,
        "validate": lambda res: "oxana" in res.lower() and "2" in res and "year" in res.lower(),
        "error_msg": "Expected a conversational string containing the product name and warranty duration."
    },
    {
        "name": "Product FAQ Vector Search",
        "query": "Is the Mini Glasis suitable for harsh Indian weather conditions?",
        "type": str,
        "validate": lambda res: "weather" in res.lower() and ("yes" in res.lower() or "designed" in res.lower() or "harsh" in res.lower() or "monsoon" in res.lower()),
        "error_msg": "Expected a conversational string answering the FAQ about weather conditions."
    },
    {
        "name": "Policy Vector Search",
        "query": "What is the return or replacement policy?",
        "type": str,
        "validate": lambda res: "return" in res.lower() or "replace" in res.lower() or "policy" in res.lower(),
        "error_msg": "Expected a conversational string explaining the return/replacement policy."
    },
    {
        "name": "Multi-Parameter Search (Structured)",
        "query": "show 3 best rated products under 2000 rupees",
        "type": list,
        "validate": lambda res: len(res) <= 3 and isinstance(res, list),
        "error_msg": "Expected a list containing up to 3 products based on rating and price."
    },
    {
        "name": "Synonym Handling Search (Structured)",
        "query": "show cheapest budget friendly gate lights",
        "type": list,
        "validate": lambda res: len(res) > 0 and isinstance(res, list),
        "error_msg": "Expected a list of budget products, verifying synonym handling."
    },
    {
        "name": "Out-of-Domain Rejection (Conversational)",
        "query": "Can you recommend a good laptop for gaming?",
        "type": (str, list),
        "validate": lambda res: (isinstance(res, list) and len(res) == 0) or (isinstance(res, str) and ("sorry" in res.lower() or "don't have" in res.lower() or "database" in res.lower())),
        "error_msg": "Expected the agent to return an empty list or strictly refuse answering out-of-domain questions."
    },
    {
        "name": "Human WhatsApp (Greeting + Search)",
        "query": "Hi, looking for some nice gate lights for my new home",
        "type": list,
        "validate": lambda res: len(res) > 0 and isinstance(res, list),
        "error_msg": "Expected the agent to handle the greeting and successfully return gate lights."
    },
    {
        "name": "Human WhatsApp (Damage Issue)",
        "query": "my order arrived today but the glass on one of the lamps is broken... what should i do?",
        "type": str,
        "validate": lambda res: "replace" in res.lower() or "return" in res.lower() or "damage" in res.lower() or "support" in res.lower() or "contact" in res.lower(),
        "error_msg": "Expected a conversational response detailing the damaged goods policy."
    },
    {
        "name": "Human WhatsApp (Shipping Check)",
        "query": "hey, do you guys deliver to bangalore? how long does it take?",
        "type": str,
        "validate": lambda res: "day" in res.lower() or "ship" in res.lower() or "deliver" in res.lower(),
        "error_msg": "Expected a conversational response explaining shipping timelines."
    }
]

def run_evaluation():
    print("========================================")
    print("      GraphRAG Agent Evaluation Suite   ")
    print("========================================")
    
    print("Initializing Agent...")
    initialize_agent()
    print("Initialization complete.\n")
    
    passed = 0
    total = len(GROUND_TRUTHS)
    
    for i, test in enumerate(GROUND_TRUTHS, 1):
        print(f"Test {i}/{total}: {test['name']}")
        print(f"Query: \"{test['query']}\"")
        
        start_time = time.time()
        result = ask_agent(test['query'])
        latency = time.time() - start_time
        
        print(f"Latency: {latency:.2f}s")
        print(f"Output Type: {type(result).__name__}")
        
        # Validation
        is_pass = False
        if not isinstance(result, test['type']):
            print(f"[FAIL] Expected type {test['type'].__name__}, got {type(result).__name__}")
        else:
            try:
                if test['validate'](result):
                    is_pass = True
                else:
                    print(f"[FAIL] {test['error_msg']}")
                    if isinstance(result, str):
                        print(f"   Returned: {result[:200]}...")
                    else:
                        print(f"   Returned: {result}")
            except Exception as e:
                print(f"[FAIL] Exception during validation: {e}")
        
        if is_pass:
            passed += 1
            print("[PASS]")
        
        print("-" * 40)
        
    print("\n========================================")
    print(f"              SUMMARY                   ")
    print(f"  Passed: {passed} / {total} ({(passed/total)*100:.1f}%)")
    print("========================================")
    
    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()
