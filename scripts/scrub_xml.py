import xml.etree.ElementTree as ET
from pathlib import Path

from tunamelt import REPO_PATH


def remove_fields_from_xml(input_file, output_file, fields_to_remove):
    print("Processing", input_file)
    tree = ET.parse(input_file)
    root = tree.getroot()

    for field in fields_to_remove:
        for elem in root.findall(f".//{field}"):
            elem.text = ""
    tree.write(output_file)


def single_test():
    input_file = "390000.xml"
    output_file = "390000-scrubbed.xml"
    fields_to_remove = ["url", "username", "email"]  # Add the fields you want to remove

    remove_fields_from_xml(input_file, output_file, fields_to_remove)

    return None


def batched_test():

    labels_dir = Path(__file__).parents[1] / "data/PNNL-TUNAMELT/labels"
    print("Labels directory:", labels_dir)
    for input_file in labels_dir.glob("**/*.xml"):
        print("XML File:", input_file)
        remove_fields_from_xml(input_file, input_file, ["url", "username", "email"])


if __name__ == "__main__":
    batched_test()
