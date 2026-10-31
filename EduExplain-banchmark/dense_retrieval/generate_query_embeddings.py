import json 
import numpy as np 
from FlagEmbedding import BGEM3FlagModel 
from pathlib import Path 
from multiprocessing import freeze_support 

SCRIPT_DIR = Path(__file__).resolve().parent 
json_queries = ( "./query_generation/main_queries.json" ) 
json_query_ids = SCRIPT_DIR / "query_ids.json" 
json_output = SCRIPT_DIR / "query_embeddings.npy" 

def main(): 
    
    # ------------------------- # Load Model # ------------------------- 
    model = BGEM3FlagModel( "~/models/bge-m3", use_fp16=True ) 
    
    # ------------------------- # Load Queries # ------------------------- 
    with open(json_queries, "r", encoding="utf-8") as f: 
        queries = json.load(f) 
        
    query_ids = [] 
    query_texts = [] 
    instruction = ( "Represent this sentence for retrieving relevant " "educational resources: " ) 

    for q in queries: 
        query_ids.append(q["query_id"]) 
        query_texts.append(instruction + q["query"]) 

    print(f"Loaded {len(query_texts)} queries.") 
    
    # ------------------------- # Encode Queries # ------------------------- 
    query_embeddings = model.encode( query_texts, batch_size=64, max_length=512 )["dense_vecs"] 
    
    # ------------------------- # Save Embeddings # ------------------------- 
    np.save(json_output, query_embeddings) 
    
    with open(json_query_ids, "w", encoding="utf-8") as f: 
        json.dump(query_ids, f, indent=2) 

    print(f"Saved embeddings to: {json_output}") 
    
    print(f"Saved query IDs to: {json_query_ids}") 
    
    print(f"Embedding shape: {query_embeddings.shape}") 
    
if __name__ == "__main__": 
    freeze_support() 
    main()
