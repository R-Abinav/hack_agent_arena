from appworld import AppWorld

task_id = "024c982_1"

print(f"Solving task {task_id} manually to demonstrate process...")

with AppWorld(task_id=task_id) as world:
    print(f"Supervisor: {world.task.supervisor['email']}")
    
    code = """
# Step 1: Login to Venmo
venmo_creds = [c for c in apis.supervisor.show_account_passwords() if c['account_name'] == 'venmo'][0]
login_res = apis.venmo.login(username="joyce-weav@gmail.com", password=venmo_creds['password'])
access_token = login_res['access_token']
print("Logged into Venmo")

# Step 2: Find Stacy
friends = apis.venmo.search_friends(access_token=access_token, query="Stacy")
stacy_email = friends[0]['email']
print("Found Stacy's email:", stacy_email)

# Step 3: Request $13 publicly
res = apis.venmo.create_payment_request(
    access_token=access_token,
    user_email=stacy_email,
    amount=13.0,
    description="For yesterday's meal",
    private=False
)
print("Payment request created:", res)

# Step 4: Complete Task
apis.supervisor.complete_task(answer=None)
"""
    output = world.execute(code)
    print("Execution Output:\n", output)
    print("Task Completed?", world.task_completed())
