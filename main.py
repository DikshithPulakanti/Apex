import os
from dotenv import load_dotenv

load_dotenv()

app_name = os.getenv('APP_NAME', 'APEX')
env = os.getenv('ENVIRONMENT', 'development')

print(f'🚀 {app_name} starting in {env} mode')
print('Docker is working correctly!')