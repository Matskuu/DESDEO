import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import argparse
import ast
import fiona
import io
import json
import requests
import subprocess
import sys
import zipfile
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.geometry as geom
import shapely.wkt as wkt

from xml.etree import ElementTree as ET


class PipelineError(Exception):
    """."""

def parse_real_estate_id(original_id: str) -> str:
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


def coord_to_polygon(coords: list[float]) -> str:
    polygon = "POLYGON (("
    for pair in coords:
        polygon = polygon + str(pair[0]) + " " + str(pair[1]) + ", "

    return polygon[:-2] + "))"


def xml_to_dict(element: ET.Element):
    # Create a dictionary to store the result
    result = {}

    # If the element has attributes, add them to the result
    if element.attrib:
        result.update(('@' + k, v) for k, v in element.attrib.items())

    # If the element has children, recursively process them
    if element:
        # Group children by tag name
        for child in element:
            child_dict = xml_to_dict(child)
            # If multiple children with the same tag, put them in a list
            if child.tag in result:
                if isinstance(result[child.tag], list):
                    result[child.tag].append(child_dict)
                else:
                    result[child.tag] = [result[child.tag], child_dict]
            else:
                result[child.tag] = child_dict
    # If the element has text, add it to the result (strip extra whitespace)
    elif element.text:
        result = element.text.strip()
    return result


def get_real_estate_polygon(realestateid: str, api_key: str):
    r = requests.get(f"https://avoin-paikkatieto.maanmittauslaitos.fi/kiinteisto-avoin/simple-features/v3/collections/PalstanSijaintitiedot/items?kiinteistotunnus={realestateid}",
                 params={"api-key": api_key, "crs": "http://www.opengis.net/def/crs/EPSG/0/3067"})

    estate_data = json.loads(r.content)
    features = estate_data["features"]
    coordinates = []
    for feature in features:
        coordinates.append(feature["geometry"]["coordinates"][0])
    return coordinates, features


def write_real_estate_xml(coordinates: list, realestateid: str) -> tuple[list[str], list]:
    error_messages = []
    coordinates_copy = coordinates.copy()
    number = 1
    for i in range(len(coordinates)):
        geometry = coordinates[i]
        polygon = coord_to_polygon(geometry)
        req = requests.post("https://avoin.metsakeskus.fi/rest/mvrest/FRStandData/v1/ByPolygon", data={"wktPolygon": polygon, "stdVersion": "MV1.9"})
        xml = req.content
        if "MV-kuvioita ei löytynyt." in xml.decode():
            error_messages.append(f"No forest found for a polygon from estate {realestateid}.")
            coordinates_copy.pop(i)
            continue
        with Path.open(f"{realestateid}/output_{number}.xml", "wb") as file:
            file.write(xml)
        number = number + 1
    """if len(coordinates) == 1:
        geometry = coordinates[0]
        polygon = coord_to_polygon(geometry)
        req = requests.post("https://avoin.metsakeskus.fi/rest/mvrest/FRStandData/v1/ByPolygon", data={"wktPolygon": polygon, "stdVersion": "MV1.9"})
        xml = req.content
        with Path.open(f"{realestateid}/output.xml", "wb") as file:
            file.write(xml)
    else:
        number = 1
        for i in range(len(coordinates)):
            geometry = coordinates[i]
            polygon = coord_to_polygon(geometry)
            req = requests.post("https://avoin.metsakeskus.fi/rest/mvrest/FRStandData/v1/ByPolygon", data={"wktPolygon": polygon, "stdVersion": "MV1.9"})
            xml = req.content
            if "MV-kuvioita ei löytynyt." in xml.decode():
                error_messages.append(f"No forest found for a polygon from estate {realestateid}.")
                coordinates_copy.pop(i)
                continue
            with Path.open(f"{realestateid}/output_{number}.xml", "wb") as file:
                file.write(xml)
            number = number + 1"""
    return error_messages, coordinates_copy


def get_polygon_dict(root: ET.ElementTree):
    orig_polygons = {}
    # make a dict of stands' coordinates
    for child in root:
        if child.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stands":
            for stand in child:
                if stand.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stand":
                    stand_id = stand.attrib["id"]
                    for s in stand.iter("{http://www.opengis.net/gml}LinearRing"):
                        if s.tag == "{http://www.opengis.net/gml}LinearRing":
                            for i in s:
                                coords = i.text.split(" ")
                                coordinate_pairs = []
                                for coordinate in coords:
                                    coordinate_pairs.append((float(coordinate.split(",")[0]), float(coordinate.split(",")[1])))
                    orig_polygons[stand_id] = coordinate_pairs
    return orig_polygons


def get_ids_to_remove(coordinates: list, realestateid: str, plot: bool = False) -> list[str]:
    for i in range(len(coordinates)):
        tree = ET.parse(f"{realestateid}/output_{i+1}.xml")
        root = tree.getroot()


        target = geom.Polygon(coordinates[i]) # when multiple holdings, go through this in a loop?

        buffer_distance = 10
        #buffer = target.buffer(buffer_distance)

        orig_polygons = get_polygon_dict(root)

        polygons = [(p, geom.Polygon(orig_polygons[p])) for p in orig_polygons]  # List of original polygons
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

        tree.write(f"{realestateid}/output_{i+1}.xml")

        if plot:
            # Plot the remaining polygons and the buffer
            _, ax = plt.subplots()

            gdf_target.plot(ax=ax, color="red", alpha=0.2)

            # Plot the original polygons (before removal) in green
            gdf_polygons.plot(ax=ax, color="green", alpha=0.5,edgecolor="black")

            # Plot the buffer area in blue
            gpd.GeoSeries([buffer]).plot(ax=ax, color="blue", alpha=0.3)

            for r in removed:
                x, y = r.geometry.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="black")

            ax.set_title('Polygons with Buffer (Removed Neighbors)')
            plt.savefig(f"{realestateid}/stands_{i+1}.png")
    """if len(coordinates) == 1:
        tree = ET.parse(f"{realestateid}/output.xml")
        root = tree.getroot()

        target = geom.Polygon(coordinates[0]) # when multiple holdings, go through this in a loop?

        buffer_distance = 10
        #buffer = target.buffer(buffer_distance)

        orig_polygons = orig_polygons = get_polygon_dict(root)

        polygons = [(p, geom.Polygon(orig_polygons[p])) for p in orig_polygons]  # List of original polygons
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

        tree.write(f"{realestateid}/output.xml")

        #print(gdf_polygons.count())

        if plot:
            # Plot the remaining polygons and the buffer
            _, ax = plt.subplots()

            gdf_target.plot(ax=ax, color="red", alpha=0.2)

            # Plot the original polygons (before removal) in green
            gdf_polygons.plot(ax=ax, color="green", alpha=0.5,edgecolor="black")

            # Plot the buffer area in blue
            gpd.GeoSeries([buffer]).plot(ax=ax, color="blue", alpha=0.3)

            for r in removed:
                x, y = r.geometry.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="black")

            ax.set_title('Polygons with Buffer (Removed Neighbors)')
            plt.savefig(f"{realestateid}/stands.png")

    else:
        for i in range(len(coordinates)):
            tree = ET.parse(f"{realestateid}/output_{i+1}.xml")
            root = tree.getroot()


            target = geom.Polygon(coordinates[i]) # when multiple holdings, go through this in a loop?

            buffer_distance = 10
            #buffer = target.buffer(buffer_distance)

            orig_polygons = get_polygon_dict(root)

            polygons = [(p, geom.Polygon(orig_polygons[p])) for p in orig_polygons]  # List of original polygons
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

            tree.write(f"{realestateid}/output_{i+1}.xml")

            if plot:
                # Plot the remaining polygons and the buffer
                _, ax = plt.subplots()

                gdf_target.plot(ax=ax, color="red", alpha=0.2)

                # Plot the original polygons (before removal) in green
                gdf_polygons.plot(ax=ax, color="green", alpha=0.5,edgecolor="black")

                # Plot the buffer area in blue
                gpd.GeoSeries([buffer]).plot(ax=ax, color="blue", alpha=0.3)

                for r in removed:
                    x, y = r.geometry.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc="black")

                ax.set_title('Polygons with Buffer (Removed Neighbors)')
                plt.savefig(f"{realestateid}/stands_{i+1}.png")"""
    return removed_ids


def write_updated_xml(tree: ET.ElementTree, realestateid: str, removed_ids: list[str]):
    root = tree.getroot()
    # remove the stands outside the real estate's polygon
    for child in root:
        if child.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stands":
            for stand in child:
                if stand.tag == "{http://standardit.tapio.fi/schemas/forestData/Stand}Stand" and stand.attrib["id"] in removed_ids:
                    child.remove(stand)
    tree.write(f"{realestateid}/output.xml")


def run_metsi(realestate_id: str): # make it a directory?
    # Run the metsi simulator with the data in the XML file
    # Requires that the following are found in the current repository:
    #   1. data directory from metsi (that has information about prices etc.)
    #   2. a control.yaml file that has the parameters for the metsi simulation

    # TODO: change to original xml, no need for copies if everything works like it should
    subprocess.run(f"metsi ./{realestateid}/output.xml ./{realestateid}")


def convert_sim_output_to_csv(realestateid: str): # make it a directory?
    # run the R script to convert the simulation output to csv for optimization purposes
    res = subprocess.run(f"Rscript ./convert2opt.R ./{realestateid}", capture_output=True)
    print(res)


def write_trees_json(realestateid: str):
    # run a python script to convert trees.txt into a more usable format
    res = subprocess.run(f"python desdeo/utopia_stuff/write_trees_json.py -d ./{realestateid}", capture_output=True)
    print(res)


def write_carbon_json(realestateid: str):
    # compute CO2 and write them into a json file to be used to form an optimization problem
    res = subprocess.run(f"python desdeo/utopia_stuff/write_carbon_json.py -d ./{realestateid}", capture_output=True)
    print(res)


def combine_xmls(realestateid: str, coordinates: list):
    if len(coordinates) != 1:
        with Path.open(f"{realestateid}/output.xml", "w") as file:
            with Path.open(f"{realestateid}/output_1.xml", "r") as file2:
                content = file2.read()
            file.write("\n".join(content.splitlines()[:-2]) + "\n")
            for i in range(1, len(coordinates)-1):
                with Path.open(f"{realestateid}/output_{i+1}.xml", "r") as file2:
                    content = file2.read()
                file.write("\n".join(content.splitlines()[2:-2]) + "\n")
            with Path.open(f"{realestateid}/output_{len(coordinates)}.xml", "r") as file2:
                content = file2.read()
            file.write("\n".join(content.splitlines()[2:]))
    else:
        with Path.open(f"{realestateid}/output.xml", "w") as file:
            with Path.open(f"{realestateid}/output_1.xml", "r") as file2:
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
    parser.add_argument("-n", dest="name", help="name of forest owner", type=str, default="test")
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    ids = args.ids
    name = args.name

    if not Path(f"{name}").is_dir():
        Path(f"{name}").mkdir()

    map_data = {}
    features = []
    for i in range(len(ids)):
        holding = i + 1
        realestateid = ids[i]
        """realestateid_mml = parse_real_estate_id(ids[i])

        if not Path(f"{realestateid}").is_dir():
            Path(f"{realestateid}").mkdir()

        coordinates, estate_data = get_real_estate_polygon(realestateid_mml, api_key)

        errors, coordinates = write_real_estate_xml(coordinates, realestateid)
        if len(errors) > 0:
            for error in errors:
                print(error)

        _ = get_ids_to_remove(coordinates, realestateid, plot=True)

        combine_xmls(realestateid, coordinates)"""
        print(len([node for _, node in ET.iterparse(f"{realestateid}/output_2.xml", events=["start-ns"])]))

        # convert the updated xml into a multiobjective optimization problem
        """run_metsi(realestateid)
        convert_sim_output_to_csv(realestateid)
        write_trees_json(realestateid)
        write_carbon_json(realestateid)

        tree = ET.parse(f"{realestateid}/output.xml")
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
    with Path.open(f"{name}/{name}.json", "w") as file:
        json.dump(map_data, file)"""

    """tree = ET.parse(f"{realestateid}/output.xml")
    root = tree.getroot()

    orig_polygons = get_polygon_dict(root)
    #print(orig_polygons)

    removed_ids = get_ids_to_remove(coordinates, orig_polygons, realestateid, plot=True)
    #print(removed_ids)

    write_updated_xml(tree, realestateid, removed_ids)"""
