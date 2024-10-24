import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    unique_ids = df['id'].unique()
    distance_matrix = pd.DataFrame(0.0, index=unique_ids, columns=unique_ids)
    for index, row in df.iterrows():
        id_a = row['id']
        id_b = row ['connected_id']
        distance = row['distance']
        distance_matrix.at[id_a,id_b]= distance
        distance_matrix.at[id_b,id_a]= distance 
        for k in unique_ids:
            for i in unique_ids:
                for j in unique_ids:
                    if distance_matrix.at[i,k]+ distance_matrix.at[k,j]< distance_matrix.at[i,j]:
                        distance_matrix.at[i,j] = distance_matrix.at[i,k]+ distance_matrix.at[k,j]
                        return distance_matrix 
    return df



import pandas as pd 
def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data =[]
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start,'id_end': id_end, 'distance': distance})
    unrolled_df =pd.DataFrame(unrolled_data)
    return unrolled_df

    return df



import pandas as pd 
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_distances =df [df['id_start']== reference_id]['distance']
    average_distance = reference_distances.mean()
    
    lower_threshold = average_distance* 0.9
    upper_threshold = average_distance* 1.1
    
    filtered_ids = df.groupby('id_start')['disatnce'].mean().reset_index()
    filtered_ids = filtered_ids[(filtered_ids['disatance'] >= lower_threshold)&
                                (filtered_ids['distance'] <= upper_threshold)]
    sorted_result = filtered_ids.sort_values(by='id_start')
    return sorted_result

    return df



import pandas as pd 
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rates ={
        'moto':0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus':2.2,
        'truck':3.6
    }
    for vehicle, in rates:
     df[vehicle]= df['distance']* rates[vehicle]

    return df



import pandas as pd 
import datetime 

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    week_day_discounts ={
        'morning': 0.8,
        'day': 1.2 ,
        'evening': 0.8
    }
weekend_discount = 0.7 
days = ["Monday","Tuesday","Wednesday","Thrusday","Friday","Saturday","Sunday"]
new_rows =[]
for_, row in df.iterrows():
id_start= row['id_start']
id_end = row['id_end']
distance =row ['disatnce']
for day in days [:5]:
 new_rows.append([id_start, id_end, distance, day, datetime.time (0,0), day, datetime.time(10,0), distance*weekday_discounts['morning']])
 new_rows.append([id_start, id_end, distance, day, datetime.time (10,0), day, datetime.time(18,0), distance*weekday_discounts['day']])
 new_rows.append([id_start, id_end, distance, day, datetime.time (18,0), day, datetime.time(23,59), distance*weekday_discounts['evening']])
for day in days[5:]:
    new_rows.append([id_start, id_end, distance, day,datetime.time(0,0),day,datetime.time(23,59),distance*weekend_discount])
    new_df =pd.DataFrame(new_rows, columns=['id_start','id_end','distance', 'start_day','start_time','end_day','end_time''toll_rate'])

return new_df
