import json
import os
import hashlib
import argparse

def hash_challenge(challenge):
    challenge_json = json.dumps(challenge, sort_keys=True)
    hash_object = hashlib.sha256(challenge_json.encode())
    return hash_object.hexdigest()

def process_challenges(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as f:
                challenge = json.load(f)
            
            challenge_hash = hash_challenge(challenge)
            output_file = os.path.join(output_dir, f"{challenge_hash}.json")
            print(f"Processing: {filename}")
            
            if not os.path.exists(output_file):
                with open(output_file, "w") as f:
                    json.dump(challenge, f, indent=4)
                print(f"Saved: {output_file}")
            else:
                print(f"Skipped (duplicate): {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate challenge files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save deduplicated JSON files")
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    process_challenges(args.input_dir, args.output_dir)