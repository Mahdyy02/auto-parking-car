import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
print(client.listVehicles())
