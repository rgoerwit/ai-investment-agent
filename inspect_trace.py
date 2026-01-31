import json


def inspect_trace(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    trace = data.get("trace", {})
    observations = trace.get("observations", [])

    print(f"Total observations: {len(observations)}")

    auditor_observations = []
    for obs in observations:
        if obs.get("name") == "Auditor" or obs.get("name") == "global_forensic_auditor":
            auditor_observations.append(obs)

    print(f"Auditor observations found: {len(auditor_observations)}")

    def print_tree(obs_id, level=0):
        children = [o for o in observations if o.get("parentObservationId") == obs_id]
        for child in children:
            indent = "  " * level
            name = child.get("name")
            type_ = child.get("type")

            # Extract Input
            input_data = child.get("input")
            input_str = ""
            if isinstance(input_data, str):
                input_str = input_data[:300]
            elif isinstance(input_data, dict):
                # Try to find relevant input fields
                if "messages" in input_data:
                    # Usually a list of messages
                    msgs = input_data["messages"]
                    if isinstance(msgs, list) and len(msgs) > 0:
                        last_msg = msgs[-1]
                        if isinstance(last_msg, dict):
                            input_str = (
                                f"Last Msg: {str(last_msg.get('content', ''))[:200]}"
                            )
                        else:
                            input_str = f"Last Msg: {str(last_msg)[:200]}"
                elif "tool_input" in input_data:
                    input_str = f"Tool Input: {input_data['tool_input']}"
                elif "input" in input_data:
                    input_str = f"Input: {str(input_data['input'])[:200]}"
                else:
                    input_str = f"Keys: {list(input_data.keys())}"

            # Extract Output
            output_data = child.get("output")
            output_str = ""
            if isinstance(output_data, str):
                output_str = output_data[:300]
            elif isinstance(output_data, dict):
                if "generations" in output_data:
                    # LLM output
                    gens = output_data["generations"]
                    if isinstance(gens, list) and len(gens) > 0:
                        first_gen = gens[0]
                        if isinstance(first_gen, list):
                            first_gen = first_gen[0]  # Sometimes nested

                        text = first_gen.get("text", "")
                        msg = first_gen.get("message", {})
                        kwargs = msg.get("additional_kwargs", {})
                        tool_calls = kwargs.get("tool_calls", [])

                        output_str = f"Gen: {text[:200]}"
                        if tool_calls:
                            output_str += f" | Tool Calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}"
                elif "text" in output_data:
                    output_str = output_data["text"][:300]
                else:
                    output_str = f"Keys: {list(output_data.keys())}"

            print(f"{indent}- {name} ({type_})")
            if input_str:
                print(f"{indent}  In: {input_str}")
            if output_str:
                print(f"{indent}  Out: {output_str}")

            print_tree(child.get("id"), level + 1)

    for i, obs in enumerate(auditor_observations):
        print(f"\n--- Auditor Observation {i+1} ---")
        print_tree(obs.get("id"))


if __name__ == "__main__":
    inspect_trace("scratch/trace-bc917bdc8cb63830708c4311f105cf28.json")
