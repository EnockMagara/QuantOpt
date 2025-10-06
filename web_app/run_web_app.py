#!/usr/bin/env python3
"""
MMS Finance Web Application Launcher
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the MMS Finance web application"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    web_app_dir = Path(__file__).parent
    
    print("🚀 Starting MMS Finance Web Application...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Web app directory: {web_app_dir}")
    
    # Change to web app directory
    os.chdir(web_app_dir)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root / 'src')
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Run the Flask application
        print("✅ Starting Flask development server...")
        print("🌐 Web application will be available at: http://localhost:5000")
        print("📊 API endpoints available at: http://localhost:5000/api/")
        print("\n" + "="*60)
        print("Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        subprocess.run([
            sys.executable, 'app.py'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Web application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

