"""A python script to get forest data and metsi simulation data by just giving real estate codes of forest holdings.

The scipt takes three arguments:

-i: A list of real estate ids. For example: 111-2-34-56 999-888-7777-6666.

-d: The directory (path) where the data will be stored. If the directory does not exist, the directory will be made.

-n: A name for the forest holdings. This is used as a name for a directory to store the data for all the given
    real estates. (for now) Assumed to be one string (no spaces). Can be, for example, the lastname of the forest owner.

An example call:

    python data_pipeline.py -i 111-2-34-56 999-888-7777-6666 -d path/to/target/directory -n Lastname

With this call the script would contact Maanmittauslaitos' API and get the polygons related to the given
real estate codes. The script will then make an HTTP request to Metsäkeskus' API to get the forest
data related to the polygons.

The polygons will then be filtered to get rid of any neighboring forest stands that get passed by the
Metsäkeskus API. This is done by creating a buffer around the polygon from Maanmittauslaitos
and then looping through all the stands from Metsäkeskus to see if their polygon is completely inside
the buffered polygon of the estate. The stands that are not completely inside will be removed.
This is visualised by drawing an image of the different polygons that showcases which
stands are inside the blue bufferzone and which are outside (drawn in black) with red color representing the
original polygon of the (part of the) holding and green color indicating stands that are determined to
be inside of the bufferzone.

The script will create a directory named 'Lastname' into the directory 'path/to/target/directory'.
Into this created directory the script then creates a directory for each real estate, in this case,
two directories named '111-2-34-56' and '999-888-7777-6666'. In these holding specific directories, the
script stores all the forest data from Metsäkeskus related to the holding and all the outputs of metsi
for the different holdings. The script will also create a 'Lastname.json' file into the 'Lastname' directory
that is a json file with information needed to draw the maps in DESDEO.

NOTE: write_trees_json.py parses trees.txt that is in Jari's changed form. May need some changes with the actual
metsi form.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import shapely.geometry as geom

NS = {
        "schema_location": "http://standardit.tapio.fi/schemas/forestData ForestData.xsd",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xlink": "http://www.w3.org/1999/xlink",
        "gml": "http://www.opengis.net/gml",
        "gdt": "http://standardit.tapio.fi/schemas/forestData/common/geometricDataTypes",
        "co": "http://standardit.tapio.fi/schemas/forestData/common",
        "sf": "http://standardit.tapio.fi/schemas/forestData/specialFeature",
        "op": "http://standardit.tapio.fi/schemas/forestData/operation",
        "dts": "http://standardit.tapio.fi/schemas/forestData/deadTreeStrata",
        "tss": "http://standardit.tapio.fi/schemas/forestData/treeStandSummary",
        "tst": "http://standardit.tapio.fi/schemas/forestData/treeStratum",
        "ts": "http://standardit.tapio.fi/schemas/forestData/treeStand",
        "st": "http://standardit.tapio.fi/schemas/forestData/Stand",
        "ci": "http://standardit.tapio.fi/schemas/forestData/contactInformation",
        "re": "http://standardit.tapio.fi/schemas/forestData/realEstate",
        "default": "http://standardit.tapio.fi/schemas/forestData"
    }


class PipelineError(Exception):
    """An error class for the data pipeline."""


def parse_real_estate_id(original_id: str) -> str:
    """Get the given real estate id in the long format.

    E.g., 111-2-34-56 --> 11100200340056

    Args:
        original_id (str): The original id with hyphens.

    Returns:
        str: The id in the long format.
    """
    # TODO: this could be expanded to be able to handle ids in different formats
    realestateid = original_id
    if "-" in realestateid:
        parts = realestateid.split("-")
        first_parts = 3
        last_parts = 4
        while len(parts[0]) != first_parts:
            parts[0] = "0" + parts[0]
        while len(parts[1]) != first_parts:
            parts[1] = "0" + parts[1]
        while len(parts[2]) != last_parts:
            parts[2] = "0" + parts[2]
        while len(parts[3]) != last_parts:
            parts[3] = "0" + parts[3]
        realestateid = "".join(parts)
    return realestateid


def coordinates_to_polygon(coordinate_pairs: list) -> str:
    """Get the polygon in the correct format to call Metsäkeskus API.

    E.g., [(612.33, 7221.22), (611.53, 7222.11)] --> 'Polygon ((612.33 7221.22, 611.53 7222.11))'

    Args:
        coordinate_pairs (list): A list of coordinate pairs as tuples or lists.

    Returns:
        str: the coordinate pairs formed into the correct string form to call Metsäkeskus API.
    """
    polygon = "POLYGON (("
    for pair in coordinate_pairs:
        polygon = polygon + str(pair[0]) + " " + str(pair[1]) + ", "
    return polygon[:-2] + "))" # replace the last ", " with "))"


def get_real_estate_coordinates(realestateid: str, api_key: str) -> list:
    """Get the real estate polygon that matches the given real estate id from Maanmittauslaitos.

    Args:
        realestateid (str): The real estate ID in the long form (e.g., 11100200340056).
        api_key (str): An API key required to contact the API.

    Returns:
        list: A list of the coordinates from Maanmittauslaitos for the given real estate ID.
    """
    r = requests.get(f"https://avoin-paikkatieto.maanmittauslaitos.fi/kiinteisto-avoin/simple-features/v3/collections/PalstanSijaintitiedot/items?kiinteistotunnus={realestateid}",
                 params={"api-key": api_key, "crs": "http://www.opengis.net/def/crs/EPSG/0/3067"})

    # get the data into a dict
    estate_data = json.loads(r.content)

    # get a list of different separate "parts" of the real estate
    features = estate_data["features"]

    # get the coordinates of the different parts into a list
    coordinates = []
    for feature in features:
        coordinates.append(feature["geometry"]["coordinates"][0])
    return coordinates


def write_real_estate_xmls(coordinates: list, realestateid: str, realestate_dir: str) -> tuple[list[str], list]:
    """Get the forest data from Metsäkeskus with a list of coordinates and write the data into XML files.

    Args:
        coordinates (list): A list of lists of coordinates. Different parts of the real estate as different lists.
        realestateid (str): The real estate ID. Used only for the error messages.
        realestate_dir (str): A directory to store the forest data XMLs in.

    Returns:
        tuple[list[str], list]: A list of possible error messages and a new coordinates list.
            If a polygon does not match any stands in Metsäkeskus' database the coordinates will be removed.
    """
    error_messages = []
    # a copy of the original coordinate list to modify is necessary
    coordinates_copy = coordinates.copy()
    # the number of the current part of the estate (each part has its own XML file)
    number = 1
    # loop through the different parts of the estate
    for i in range(len(coordinates)):
        # get the polygon in the correct form to call Metsäkeskus API
        polygon = coordinates_to_polygon(coordinates[i])

        # call Metsäkeskus API to get the forest data for the polygon
        req = requests.post("https://avoin.metsakeskus.fi/rest/mvrest/FRStandData/v1/ByPolygon", data={"wktPolygon": polygon, "stdVersion": "MV1.9"})
        xml = req.content

        # if no stands are found with the polygon
        if "MV-kuvioita ei löytynyt." in xml.decode():
            # add an error message stating that for a polygon, no forest data was found
            error_messages.append(f"No forest found for a polygon from estate {realestateid}.")

            # remove the polygon from the list of coordinates
            coordinates_copy.pop(i)
            continue

        # write the forest data into an XML file
        with Path.open(f"{realestate_dir}/output_{number}.xml", "wb") as file:
            file.write(xml)

        # raise the number for the next loop
        number = number + 1
    return error_messages, coordinates_copy


def get_polygon_dict(root: ET.ElementTree) -> dict[str, dict[str, tuple[float, float]]]:
    """Get a dict of the stands' polygons from a given ElementTree.

    Args:
        root (ET.ElementTree): ElementTree with the polygons.

    Returns:
        dict[str, dict[str, tuple[float, float]]]: A dict with stand IDs as keys and a dict with the exterior and interior
            polygons as values.
    """
    orig_polygons = {}
    # loop through the children of the root
    for child in root:
        if child.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stands":
            # loop through the stands
            for stand in child:
                if stand.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stand":
                    # store the stand ID
                    stand_id = stand.attrib["id"]
                    exterior_and_interior = {}
                    # find the exterior polygon for the stand
                    for e in stand.iter("{http://www.opengis.net/gml}exterior"):
                        for s in e.iter("{http://www.opengis.net/gml}LinearRing"):
                            for i in s:
                                coordinates = i.text.split(" ")
                                coordinate_pairs = []
                                for coordinate in coordinates:
                                    coordinate_pairs.append((float(coordinate.split(",")[0]), float(coordinate.split(",")[1])))
                    exterior_and_interior["exterior"] = coordinate_pairs
                    coordinate_pairs = []
                    # if exists, find the interior polygon (a hole in the stand)
                    # TODO: what if there are multiple interiors?
                    # does iter loop through all of them?
                    for e in stand.iter("{http://www.opengis.net/gml}interior"):
                        for s in e.iter("{http://www.opengis.net/gml}LinearRing"):
                            for i in s:
                                coordinates = i.text.split(" ")
                                for coordinate in coordinates:
                                    coordinate_pairs.append((float(coordinate.split(",")[0]), float(coordinate.split(",")[1])))
                    exterior_and_interior["interior"] = coordinate_pairs
                    orig_polygons[stand_id] = exterior_and_interior
    return orig_polygons


def get_ids_to_remove(coordinates: list, realestate_dir: str, plot: bool = False) -> list[str]:
    for i in range(len(coordinates)):
        tree = ET.parse(f"{realestate_dir}/output_{i+1}.xml")
        root = tree.getroot()

        target = geom.Polygon(coordinates[i]) # when multiple holdings, go through this in a loop?

        buffer_distance = 10
        #buffer = target.buffer(buffer_distance)

        orig_polygons = get_polygon_dict(root)

        polygons = []
        for key, value in orig_polygons.items():
            if len(value["interior"]) > 0:
                polygons.append((key, geom.Polygon(value["exterior"], holes=[value["interior"]])))
            else:
                polygons.append((key, geom.Polygon(value["exterior"])))

        gdf_polygons = gpd.GeoDataFrame({"stand_id": [p[0] for p in polygons], "geometry": [p[1] for p in polygons]})

        # Create a GeoDataFrame for the target (holding) polygon
        gdf_target = gpd.GeoDataFrame(geometry=[target])

        # Buffer the target polygon
        buffer = gdf_target.buffer(buffer_distance).iloc[0]  # Create buffer of target

        removed = [] # for plotting purposes
        removed_ids = []
        # Remove any neighboring stands from the buffer
        for index, stand in gdf_polygons.iterrows():
            if not buffer.contains(stand.geometry):
                removed.append(stand) # for plotting purposes
                removed_ids.append(stand.stand_id)
                gdf_polygons = gdf_polygons.drop(index)

        for child in root:
            if child.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stands":
                to_remove = []
                for stand in child:
                    if stand.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stand" and stand.attrib["id"] in removed_ids:
                        to_remove.append(stand)
                for r in to_remove:
                    child.remove(r)

        def fix_prefixes(element, namespaces):
            # Add the namespace to each element if needed
            for child in element:
                fix_prefixes(child, namespaces)

            if element.tag.startswith("{"):
                # Extract the namespace part from the tag
                namespace = element.tag.split("}")[0][1:]
                prefix = ""
                for key, value in namespaces.items():
                    if namespace == value:
                        prefix = key
                element.tag = f"{prefix}:{element.tag.split('}')[1]}"

        # Use namespaces manually as required
        namespaces = NS
        fix_prefixes(root, namespaces)

        new_xml = ET.tostring(root).decode()

        namespaces = ""
        for key, value in NS.items():
            if value == "http://standardit.tapio.fi/schemas/forestData ForestData.xsd":
                namespaces = namespaces + f'xsi:{key}="{value}"' + " "
            elif key == "default":
                namespaces = namespaces + f'xmlns="{value}"' + " "
            else:
                namespaces = namespaces + f'xmlns:{key}="{value}"' + " "

        namespaces = namespaces + 'schemaPackageVersion="V20" schemaPackageSubversion="V20.01"'

        first_row = "<ForestPropertyData " + namespaces + ">"

        new_xml_list = new_xml.split("\n")
        new_xml_list[0] = first_row
        new_xml_list[-1] = "</ForestPropertyData>"
        new_xml = "\n".join(new_xml_list)
        with Path.open(f"{realestate_dir}/output_{i+1}.xml", "w") as file:
            file.write(new_xml)

        if plot:
            # Plot the remaining polygons and the buffer
            _, ax = plt.subplots()

            gdf_target.plot(ax=ax, color="red", alpha=0.2)

            gdf_polygons.plot(ax=ax, color="green", alpha=0.5,edgecolor="black")

            # Plot the buffer area in blue
            gpd.GeoSeries([buffer]).plot(ax=ax, color="blue", alpha=0.3)

            for r in removed:
                x, y = r.geometry.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="black")

            ax.set_title('Polygons with Buffer (Removed Neighbors)')
            plt.savefig(f"{realestate_dir}/stands_{i+1}.png")
    return removed_ids


def run_metsi(realestate_dir: str): # make it a directory?
    # Run the metsi simulator with the data in the XML file
    # Requires that the following are found in the current repository:
    #   1. data directory from metsi (that has information about prices etc.)
    #   2. a control.yaml file that has the parameters for the metsi simulation
    res = subprocess.run(f"metsi {realestate_dir}/output.xml {realestate_dir}", capture_output=True)
    if res.stderr:
        raise PipelineError(msg="Error when running metsi: " + res.stderr.decode())


def convert_sim_output_to_csv(realestate_dir: str): # make it a directory?
    # run the R script to convert the simulation output to csv for optimization purposes
    # TODO: hardcode the correct location of the R file or add that as an argument to this python script
    res = subprocess.run(f"Rscript ./convert2opt.R {realestate_dir}", capture_output=True)
    if res.stderr:
        raise PipelineError(msg="Error converting simulation data to usable CSV: " + res.stderr.decode())


def write_trees_json(realestate_dir: str):
    # run a python script to convert trees.txt into a more usable format
    # TODO: hardcode the correct location of the write_trees_json.py file or add that as an argument to this python script
    res = subprocess.run(f"python desdeo/utopia_stuff/write_trees_json.py -d {realestate_dir}", capture_output=True)
    if res.stderr:
        raise PipelineError(msg="Error when writing trees.json: " + res.stderr.decode())


def write_carbon_json(realestate_dir: str):
    # compute CO2 and write them into a json file to be used to form an optimization problem
    # TODO: hardcode the correct location of the write_carbon_json.py file or add that as an argument to this python script
    res = subprocess.run(f"python desdeo/utopia_stuff/write_carbon_json.py -d {realestate_dir}", capture_output=True)
    if res.stderr:
        raise PipelineError(msg="Error when writing carbon.json: " + res.stderr.decode())


def combine_xmls(realestate_dir: str, coordinates: list):
    if len(coordinates) != 1:
        with Path.open(f"{realestate_dir}/output.xml", "w") as file:
            with Path.open(f"{realestate_dir}/output_1.xml", "r") as file2:
                content = file2.read()
            file.write("\n".join(content.splitlines()[:-2]) + "\n")
            for i in range(1, len(coordinates)-1):
                with Path.open(f"{realestate_dir}/output_{i+1}.xml", "r") as file2:
                    content = file2.read()
                file.write("\n".join(content.splitlines()[2:-2]) + "\n")
            with Path.open(f"{realestate_dir}/output_{len(coordinates)}.xml", "r") as file2:
                content = file2.read()
            file.write("\n".join(content.splitlines()[2:]))
    else:
        with Path.open(f"{realestate_dir}/output.xml", "w") as file:
            with Path.open(f"{realestate_dir}/output_1.xml", "r") as file2:
                content = file2.read()
            file.write(content)


if __name__ == "__main__":
    path_to_api_key = "C:/MyTemp/code/UTOPIA/DESDEO/key.txt" # TODO: make this an argument as well
    with Path.open(f"{path_to_api_key}", "r") as f:
        api_key = f.read()

    # TODO: add an argument that takes the directory to save all the files to
    parser = argparse.ArgumentParser()
    arg_msg = "Real estate ids as a list. For example: 111-2-34-56 999-888-7777-6666."
    parser.add_argument("-i", dest="ids", help=arg_msg, type=str, nargs="*", default=[])
    parser.add_argument("-d", dest="dir", help="target directory for data", type=str)
    parser.add_argument("-n", dest="name", help="name of forest owner", type=str, default="test")
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    ids = args.ids
    target_dir = args.dir
    name = args.name

    if not Path(f"{target_dir}").is_dir():
        Path(f"{target_dir}").mkdir()

    if not Path(f"{target_dir}/{name}").is_dir():
        Path(f"{target_dir}/{name}").mkdir()

    map_data = {}
    features = []
    for i in range(len(ids)):
        holding = i + 1
        realestateid = ids[i]
        realestateid_mml = parse_real_estate_id(ids[i])
        realestate_dir = f"{target_dir}/{name}/{realestateid}"

        if not Path(realestate_dir).is_dir():
            Path(realestate_dir).mkdir()

        coordinates = get_real_estate_coordinates(realestateid_mml, api_key)

        errors, coordinates = write_real_estate_xmls(coordinates, realestateid, realestate_dir)
        if len(errors) > 0:
            for error in errors:
                print(error)

        _ = get_ids_to_remove(coordinates, realestate_dir, plot=True)

        combine_xmls(realestate_dir, coordinates)

        # convert the updated xml into a multiobjective optimization problem
        run_metsi(realestate_dir)
        convert_sim_output_to_csv(realestate_dir)
        write_trees_json(realestate_dir)
        write_carbon_json(realestate_dir)

        tree = ET.parse(f"{realestate_dir}/output.xml")
        root = tree.getroot()

        for child in root:
            if child.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stands":
                for stand in child:
                    if stand.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stand":
                        feature = {}
                        properties = {}
                        geometry = {}
                        geometry["type"] = "Polygon"
                        stand_id = stand.attrib["id"]
                        properties["id"] = int(stand_id)
                        properties["estate_code"] = realestateid
                        coordinates = []
                        properties["number"] = int(stand.find("{http://standardit.tapio.fi/schemas/forestData/Stand}StandBasicData").find("{http://standardit.tapio.fi/schemas/forestData/Stand}StandNumber").text)
                        for s in stand.iter("{http://www.opengis.net/gml}LinearRing"):
                            if s.tag == "{http://www.opengis.net/gml}LinearRing":
                                coordinate_pairs = []
                                for i in s:
                                    coords = i.text.split(" ")
                                    coordinate_pairs = []
                                    for coordinate in coords:
                                        coordinate_pairs.append([float(coordinate.split(",")[0]), float(coordinate.split(",")[1])])
                                    coordinates.append(coordinate_pairs)
                        geometry["coordinates"] = coordinates
                        feature["properties"] = properties
                        feature["geometry"] = geometry
                        features.append(feature)
    map_data["features"] = features
    with Path.open(f"{target_dir}/{name}/{name}.json", "w") as file:
        json.dump(map_data, file)
