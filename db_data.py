from pymongo.mongo_client import MongoClient

# Your MongoDB connection string
uri = "mongodb+srv://kareemalchorbaji_db_user:Admin23@cluster0.12offo8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

def debug_mongodb():
    try:
        client = MongoClient(uri)
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully!")
        
        # List ALL databases
        all_databases = client.list_database_names()
        print(f"\n📋 ALL DATABASES IN YOUR CLUSTER:")
        for i, db_name in enumerate(all_databases, 1):
            print(f"  {i}. {db_name}")
        
        # Check specifically for AI database
        print(f"\n🔍 LOOKING FOR 'AI' DATABASE:")
        if 'AI' in all_databases:
            print("✅ 'AI' database EXISTS!")
            
            # Get AI database
            ai_db = client['AI']
            collections = ai_db.list_collection_names()
            print(f"📁 Collections in 'AI' database: {collections}")
            
            if 'NetworkData' in collections:
                network_collection = ai_db['NetworkData']
                doc_count = network_collection.count_documents({})
                print(f"📊 Documents in 'NetworkData' collection: {doc_count}")
                
                if doc_count > 0:
                    print("\n📄 SAMPLE DOCUMENT:")
                    sample_doc = network_collection.find_one()
                    print(sample_doc)
                else:
                    print("❌ NetworkData collection exists but is EMPTY")
            else:
                print("❌ 'NetworkData' collection NOT FOUND in AI database")
        else:
            print("❌ 'AI' database NOT FOUND")
            print("   This means your Python script hasn't run successfully yet.")
        
        # Also check what's in sample_mflix for comparison
        print(f"\n🔍 SAMPLE_MFLIX DATABASE INFO:")
        mflix_db = client['sample_mflix']
        mflix_collections = mflix_db.list_collection_names()
        print(f"📁 Collections in sample_mflix: {mflix_collections}")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def check_local_file():
    import os
    print(f"\n📂 LOCAL FILE CHECK:")
    print(f"Current directory: {os.getcwd()}")
    
    # Check different possible file paths
    possible_paths = [
        "Network_Data/phisingData.csv",
        "Network_Data\\phisingData.csv", 
        "phisingData.csv",
        "data/phisingData.csv"
    ]
    
    found_file = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found CSV file at: {path}")
            size = os.path.getsize(path)
            print(f"   File size: {size} bytes")
            found_file = True
            break
    
    if not found_file:
        print("❌ CSV file NOT FOUND in any expected location")
        print("   Available files in current directory:")
        try:
            files = os.listdir('.')
            for f in files[:10]:  # Show first 10 files
                print(f"   - {f}")
        except:
            print("   Could not list files")

if __name__ == "__main__":
    print("🚀 MONGODB & FILE DEBUG REPORT")
    print("=" * 50)
    
    debug_mongodb()
    check_local_file()
    
    print(f"\n💡 NEXT STEPS:")
    print("1. If you see 'AI' database above, click on it in MongoDB Atlas")
    print("2. If you don't see 'AI' database, your Python script hasn't run successfully")
    print("3. If CSV file not found, create Network_Data folder and put your CSV there")
    print("4. Run your data_push.py script again")