from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    for i in range(0, len(lst),n):
        left, right = i, min(i+n-1, len(lst)-1)
        while left < right :
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1

    return lst
print(reverse_by_n_elements([1,2,3,4,5,6,7,8], 3))
print(reverse_by_n_elements([1,2,3,4,5], 2))
print(reverse_by_n_elements([10,20,30,40,50,60,70], 4))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result ={}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result [length].append(string)
    
    return dict(sorted(result.items()))
print(group_by_length(["apple","bat","car","elephant","dog","bear"]))
print(group_by_length(["one","two","three","four"]))
print(group_by_length(["short","medium","longer","tiny","huge"]))




def flatten_dict(nested_dict: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items ={}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key 
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key,sep))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{i}]",sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = value
    return items 
#example input for testing 
nested_input = {
    "road": {
        "name": "highway 1",
        "length" : 350,
        "sections" :[
            {
                "id":1,
                "condition":{
                    "pavement":"good",
                    "traffic" :"moderate"
                }
            }
        ]
    }
}

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start,len(nums)):
            if nums[i] in seen:
                continue 
            seen.add(nums[i])
            nums[start], nums[i] =nums[i], nums[start]
            backtrack(start +1)
            nums[start],nums[i]= nums[i], nums[start]
    result=[]
    nums.sort()
    backtrack(0)
    return result
    pass
 # example 
input_list =[1,1,2]
print(unique_permutations(input_list))


import re
from typing import list  
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',
        r'\b(0[1-9]|1[0-2]/(0[1-9]|[12][0-9]|3[01])-(\d{4})\b)',
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'
    ]
    combined_pattern ='|'.join(patterns)
    matches = re.findall(combined_pattern,text)
    valid_dates = []
    for match in matches:
        valid_dates.append(''.join(filter(None,match)))
    return valid_dates
text = " I was born on 23-08-1994, my friend on 06/23/1994, and another one on 1994.8.23."
print(find_all_dates(text))
pass


"""
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
     """

import math
def decode_polyline(polyline_str: str):
    if not polyline_str:
        return[]
    index,lat,lng =0,0,0
    coordinates =[]
    length = len(polyline_str)
    while index < length:
        shift, result =0,0
        while true :
            if index >= length:
                raise IndexError("polyline string is not valid.")
            byte = ord(polyline_str[index])-63 
            index+= 1
            result |= (byte & 0*1F)<< shift
            shift+=5
            if byte < 0*20:
                break 
        delta_lat = ~(result>> 1) if (result & 1) else (result >> 1)
        lat += delta_lat
        shift, result =0,0
        while true:
            if index >= length:
                raise IndexError("polyline string is not valid.")
            byte = ord(polyline_str[index]-63)
            index+= 1
            result|=(byte & 0*1F)<< shift
        shift+= 5           
        if byte < 0*20:
            break 
        delta_lng = ~(result >> 1) if (result & 1) else (result >>1)
        lng += delta_lng
        coordinates.append((lat/ 1e5, lng/ 1e5))
        return coordinates 
    def haversine(coord1, coord2):
        R =6371000
        lat1, lon1 = coord1
        lat2, lon2 = coord2 
        phi1 =math.radians(lat1)
        phi2 =math.radians(lat2)
        delta_phi = math.radians (lat2 - lat1)
        delta_lambda =math.radians (lon2 -lon1)
        a = math.sin(delta_phi/2)** 2 +math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
        c= 2* math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R*c
    def polyline_to_dataframe(polyline_str: str):
        coordinates = deocde_polyline (polyline_str)  
        data =[]
        prev_coord = None 
        for coord in coordinates :
            if prev_coord is None:
                distance =0 
            else:
                distance = haversine(prev_coord, coord)

            data.append({
                'latitude': coord[0],
                'longitude' : coord[1],
                'distance' : distance
            }) 
            prev_coord =coord 
            return data
        polyline_str = "gfo}etohhU~@zA?V{A~E_M|j@E~A?DAq@A?Wk@"
        try:
            decoded_df = polyline_to_dataframe(polyline_str)
            for row in decoded_df:
                print (row)
        except Exception as e :
            print(f"an error occured: {str(e)}")

        

from typing import list
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix =[[matrix[n-j-1][i]for j in range(n)] for i in range(n)]
    final_matrix = [[0] * n for_in range(n)]
    for i in range(n):
        row_sum = sum(rotated_matrix[i])
        col_sum = sum(rotated_matrix[k][j] for k in range(n))
        final_matrix[i][j]= row_sum + col_sum -rotated_matrix[i][j]
        return final_matrix
    input_matrix = [[1,2,3],[4,5,6],[7,8,9]]
    result = rotate_and_multiply_matrix(input_matrix)
    print(result)
    return []



"""
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
from datetime import datetime, timedelta
import csv
def time_check(df):
    time_data={}
    for row in df:
        id = row['id']
        id_2 = row['id_2']
        start_day = row['startDay']
        start_Time = row['start_Time']
        end_Day = row['endDay']
        end_Time =row ['endTime']
        key = (id, id_2)
        if key not in time_data:
            time_data[key] ={
                'days': set(),
                'start_times': [],
                'end_times' :[]

            }
            time_data[key]['days'].add(start_day)
            time_data[key]['days'].add(end_day)
            time_data[key]['start_times'].append(datetime.strptime(start_Time,'%H:%M:%s'))
            time_data[key]['end_times'].append(datetime.strptime(end_Time,'%H:%M:%s'))
            results ={}
            for key, values in time_data.items():
                days_covered = len(values['days'])== 7
                full_day_covered = (max(values['end_times'])-min(values['start_time']))>= timedelta(hours=24)
                results[key]= not(days_covered and full_day_covered)
            return results

    
