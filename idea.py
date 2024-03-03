import numpy as np
# Example list of 2D arrays
size = 256
n = 20
arrays_list = []
images_list = []
# Generate and append n matrices to the list
for _ in range(n):
    random_matrix = np.random.randint(0, 13, size=(size, size))
    arrays_list.append(random_matrix)

for _ in range(n):
    random_matrix = np.random.randint(0, 256, size=(size, size))
    images_list.append(random_matrix)

# Initialize the dictionary for segments
segments_dict = {i: [] for i in range(13)}  # Segments 0 to 12

# Iterate over each 2D array
for array_index, segments_2d in enumerate(arrays_list):
    # Temporary storage for this array's segments to avoid appending empty lists later
    temp_storage = {i: [] for i in range(13)}
    
    for x in range(segments_2d.shape[0]):
        for y in range(segments_2d.shape[1]):
            segment_value = segments_2d[x, y]
            # Append the (x, y) coordinates to the temp storage for this segment
            temp_storage[segment_value].append([x, y])
    
    # Now, for each segment, append this array's coordinates to the main dictionary
    for segment in segments_dict:
        if temp_storage[segment]:  # Only append if there are coordinates for this segment
            segments_dict[segment].append(temp_storage[segment])

# Optionally, convert the lists of coordinates for each segment into numpy arrays
for segment in segments_dict:
    for array_index in range(len(segments_dict[segment])):
        segments_dict[segment][array_index] = np.array(segments_dict[segment][array_index])

# segments_dict now maps each segment number to a list of 2D arrays, 
# where each 2D array contains the (x, y) coordinates for that segment across the provided 2D arrays

average_pixels_per_segment = {}

# Iterate over each segment in the dictionary
for segment, arrays_list in segments_dict.items():
    total_pixels = 0  # To accumulate the total number of pixels for this segment
    contributing_arrays = 0  # To count how many arrays contributed pixels to this segment

    # Sum up the number of pixels from each 2D array for this segment
    for array in arrays_list:
        if len(array) > 0:  # Check if the array actually contributes pixels
            total_pixels += len(array)
            contributing_arrays += 1

    # Calculate the average if there are contributing arrays, else set to 0
    if contributing_arrays > 0:
        average_pixels = round(total_pixels / contributing_arrays)
    else:
        average_pixels = 0

    # Store the average number of pixels for this segment
    average_pixels_per_segment[segment] = average_pixels

# average_pixels_per_segment now contains the average number of pixels for each segment
#print(average_pixels_per_segment)

# Initialize a dictionary to hold the pixel occurrence dictionaries for each segment
pixel_occurrences_per_segment = {}

# Iterate over each segment and its list of 2D arrays containing pixel coordinates
for segment, arrays_list in segments_dict.items():
    # Initialize a dictionary to count occurrences of each pixel coordinate for this segment
    pixel_counts = {}

    # Iterate over each subarray (2D array) for this segment
    for subarray in arrays_list:
        for pixel in subarray:
            pixel_tuple = tuple(pixel)  # Convert the pixel list to a tuple to use it as a dictionary key
            if pixel_tuple in pixel_counts:
                pixel_counts[pixel_tuple] += 1  # Increment the count if the pixel is already in the dictionary
            else:
                pixel_counts[pixel_tuple] = 1  # Initialize the count for new pixels

    # Store the pixel occurrence dictionary for this segment
    pixel_occurrences_per_segment[segment] = pixel_counts

# pixel_occurrences_per_segment now contains for each segment a dictionary of pixel coordinates and their occurrences
sorted_pixel_occurrences = {}

# Iterate over each segment and its pixel occurrence dictionary
for segment, pixel_dict in pixel_occurrences_per_segment.items():
    # Sort the pixel occurrence dictionary by value (occurrence count) in descending order
    sorted_pixel_dict = {k: v for k, v in sorted(pixel_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Store the sorted dictionary for this segment
    sorted_pixel_occurrences[segment] = sorted_pixel_dict


top_pixel_keys_per_segment = {}

# Iterate over each segment and its sorted pixel occurrence dictionary
for segment, sorted_dict in sorted_pixel_occurrences.items():
    # Retrieve the average pixel count for this segment, rounded to the nearest integer
    avg_count = int(average_pixels_per_segment[segment])
    
    # Extract the keys for the top avg_count instances
    top_keys = list(sorted_dict.keys())[:avg_count]
    
    # Store the list of top keys for this segment
    top_pixel_keys_per_segment[segment] = top_keys

# top_pixel_keys_per_segment now contains lists of pixel coordinates for the top avg_count instances for each segment

# print(top_pixel_keys_per_segment)
mean_human = np.full((size,size), 500)
for segment, arrays_list in segments_dict.items():
    for coord in top_pixel_keys_per_segment[segment]:
        all_instances = []
        for i in range(len(arrays_list)):
            if coord in arrays_list[i]:
                all_instances.append(images_list[i][coord[0]][coord[1]])
        mean_human[coord[0]][[coord[1]]] = round(sum(all_instances)/len(all_instances))
print(mean_human)

for m_index, m_value in enumerate(mean_human):
    for n_index, n_value in enumerate(m_value):
        if n_value == 500:
            summa = 0
            count = 0
            for j in range(max(0, m_index-1), min(len(mean_human), m_index+2)):
                for i in range(max(0, n_index-1), min(len(m_value), n_index+2)):
                    if mean_human[j][i] != 500:
                        count += 1
                        summa += mean_human[j][i]

            if count > 0:
                mean_human[m_index][n_index] = round(summa / count)

print(mean_human)