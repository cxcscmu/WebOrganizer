import yaml

def get_excel_col(n: int) -> str:
    """
    Converts a positive integer to an Excel-style column name.
    1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, 28 -> AB, ...
    """
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name

# Path to the YAML file
file_path = '/home/gonilude/WebOrganizer/define_domains/taxonomies/skills.yaml'

try:
    # Load the YAML file
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the required parts
    system_template = config.get('system_template', '')
    template = config.get('template', '')
    choices_list = config.get('choices', [])
    demonstrations = config.get('demonstrations', [])

    # Format the choices with Excel-style column letters (A, B, ..., AA, AB, ...)
    # This handles the 48 choices in skills.yaml correctly.
    formatted_choices = "\n".join(
        f"{get_excel_col(i + 1)}: {choice.strip()}" for i, choice in enumerate(choices_list)
    )


    # Get URL and text from the first demonstration
    if demonstrations:
        demo = demonstrations[4]
        url = demo.get('url', 'N/A')
        text = demo.get('text', 'N/A')
    else:
        url = "http://example.com"
        text = "This is example content."

    # Create a dictionary of format arguments
    format_args = {
        'choices': formatted_choices,
        'url': url,
        'text': text
    }

    # Populate the system_template
    populated_system_template = system_template.format(**format_args)

    # Populate the main template
    populated_template = template.format(**format_args)

    # --- Print the results ---
    with open("system_template.txt", "w") as f:
        f.write(populated_system_template)

    with open("template.txt", "w") as f:
        f.write(populated_template)

    # print("\n")
    # print("--- Populated System Template ---")
    # print(populated_system_template)
    # print("\n" + "="*80 + "\n")
    # print("--- Populated Template ---")
    # print(populated_template)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
