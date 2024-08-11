from organizations import clients, ORGS_ID, ALGO_ORG_ID, DATA_PROVIDER_ORGS_ID
import prepare_data

def main():
    print(f"Number of organizations: {len(ORGS_ID)}")
    print(f"Algorithm provider: {ALGO_ORG_ID}")
    print(f"Data providers: {DATA_PROVIDER_ORGS_ID}")
    
    # This will execute the data preparation
    prepare_data

if __name__ == "__main__":
    main()