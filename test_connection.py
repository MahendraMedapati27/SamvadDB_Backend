#!/usr/bin/env python3
"""
Simple script to test database connection and diagnose issues.
Run this script to check if your database connection is working properly.
"""

import requests
import json
import sys

def test_backend_health():
    """Test the backend health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running")
            print(f"   Status: {data.get('status')}")
            print(f"   Database Connected: {data.get('db_connected')}")
            print(f"   Orchestrator Ready: {data.get('orchestrator_ready')}")
            print(f"   Message: {data.get('message')}")
            
            if data.get('connection_info'):
                conn_info = data['connection_info']
                print(f"   Connected to: {conn_info.get('db_type')} database '{conn_info.get('database')}' on {conn_info.get('host')}:{conn_info.get('port')}")
            
            return data.get('db_connected') and data.get('orchestrator_ready')
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure the backend server is running on http://localhost:8000")
        print("   Run: cd Backend && python start_backend.py")
        return False
    except Exception as e:
        print(f"❌ Error testing backend health: {e}")
        return False

def test_query_endpoint():
    """Test the query endpoint."""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json={"query": "test query"}
        )
        
        if response.status_code == 400:
            data = response.json()
            print(f"❌ Query endpoint returned 400: {data.get('detail', 'Unknown error')}")
            return False
        elif response.status_code == 200:
            print("✅ Query endpoint is working")
            return True
        else:
            print(f"❌ Query endpoint returned unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing query endpoint: {e}")
        return False

def main():
    print("🔍 Testing SamvadDB Connection...")
    print("=" * 50)
    
    # Test backend health
    backend_ok = test_backend_health()
    
    if not backend_ok:
        print("\n💡 To fix this issue:")
        print("1. Make sure the backend server is running:")
        print("   cd Backend && python start_backend.py")
        print("2. Connect to a database through the frontend:")
        print("   - Go to the Connections page")
        print("   - Add or select a database connection")
        print("   - Click 'Connect'")
        print("3. Then try querying again")
        return
    
    print("\n🔍 Testing query endpoint...")
    query_ok = test_query_endpoint()
    
    if query_ok:
        print("\n✅ Everything is working! You can now use the chat interface.")
    else:
        print("\n💡 The backend is running but not ready for queries.")
        print("   Please connect to a database through the frontend first.")

if __name__ == "__main__":
    main() 