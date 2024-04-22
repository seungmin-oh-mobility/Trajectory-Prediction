import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os 
from PIL import Image
import imageio

# Original Data
df = pd.read_csv('NGSIM_US101.csv')

# Function to convert Feet to Meter
def feetToMeter(value):
    return value / 3.28084

# Dataframe unit convertion
df['Local_X'] = df['Local_X'].apply(feetToMeter)
df['Local_Y'] = df['Local_Y'].apply(feetToMeter)
df['Global_X'] = df['Global_X'].apply(feetToMeter)
df['Global_Y'] = df['Global_Y'].apply(feetToMeter)
df['Vehicle_Length'] = df['Vehicle_Length'].apply(feetToMeter)
df['Vehicle_Width'] = df['Vehicle_Width'].apply(feetToMeter)
df['Vehicle_Velocity'] = df['Vehicle_Velocity'].apply(feetToMeter)
df['Vehicle_Acceleration'] = df['Vehicle_Acceleration'].apply(feetToMeter)

# Add previous lane ID data
df['Previous_Lane_ID'] = df['Lane_ID'].shift(1)
df['Previous_Lane_ID'] = np.where(df['Vehicle_ID'] != df['Vehicle_ID'].shift(1), np.nan, df['Previous_Lane_ID']) # row where vehicle ID changes

# Add movement data (0: Straight, 1: Left lane change, 2: Right lane change)
df['Movement'] = 0

for i in range(1, len(df)):
    curLaneID = df.loc[i, 'Lane_ID']
    prevLaneID = df.loc[i, 'Previous_Lane_ID']
    
    if pd.isna(prevLaneID):
        df.loc[i, 'Movement'] = 0 # Straight
    elif (curLaneID < prevLaneID) and not (curLaneID == 6 and prevLaneID == 7):
        df.loc[i, 'Movement'] = 1 # Left Lane Change
    elif (curLaneID > prevLaneID) and not (curLaneID == 8 and prevLaneID == 6):
        df.loc[i, 'Movement'] = 2 # Right Lane Change
    else: 
        df.loc[i, 'Movement'] = 0 # Straight

# Dataframe for lane change
df_LC = df[df['Movement']>0]

# Determine lane separation lines
lane1to2 = df_LC[(df_LC['Lane_ID']==2) & (df_LC['Previous_Lane_ID']==1) ]
lane2to3 = df_LC[(df_LC['Lane_ID']==3) & (df_LC['Previous_Lane_ID']==2) ]
lane3to4 = df_LC[(df_LC['Lane_ID']==4) & (df_LC['Previous_Lane_ID']==3) ]
lane4to5 = df_LC[(df_LC['Lane_ID']==5) & (df_LC['Previous_Lane_ID']==4) ]
lane5to6 = df_LC[(df_LC['Lane_ID']==6) & (df_LC['Previous_Lane_ID']==5) ]
lane5to7 = df_LC[(df_LC['Lane_ID']==7) & (df_LC['Previous_Lane_ID']==5) ]
lane5to8 = df_LC[(df_LC['Lane_ID']==8) & (df_LC['Previous_Lane_ID']==5) ]

lane2to1 = df_LC[(df_LC['Lane_ID']==1) & (df_LC['Previous_Lane_ID']==2) ]
lane3to2 = df_LC[(df_LC['Lane_ID']==2) & (df_LC['Previous_Lane_ID']==3) ]
lane4to3 = df_LC[(df_LC['Lane_ID']==3) & (df_LC['Previous_Lane_ID']==4) ]
lane5to4 = df_LC[(df_LC['Lane_ID']==4) & (df_LC['Previous_Lane_ID']==5) ]
lane6to5 = df_LC[(df_LC['Lane_ID']==5) & (df_LC['Previous_Lane_ID']==6) ]
lane7to5 = df_LC[(df_LC['Lane_ID']==5) & (df_LC['Previous_Lane_ID']==7) ]
lane8to5 = df_LC[(df_LC['Lane_ID']==5) & (df_LC['Previous_Lane_ID']==8) ]

mean_lane1and2 = (lane1to2['Local_X'].mean() + lane2to1['Local_X'].mean())/2
mean_lane2and3 = (lane2to3['Local_X'].mean() + lane3to2['Local_X'].mean())/2
mean_lane3and4 = (lane3to4['Local_X'].mean() + lane4to3['Local_X'].mean())/2
mean_lane4and5 = (lane4to5['Local_X'].mean() + lane5to4['Local_X'].mean())/2
mean_lane5and6 = (lane5to6['Local_X'].mean() + lane6to5['Local_X'].mean())/2

# Find the separation line between lanes 5 and 7 with linear trend line
lane5and7 = pd.concat([lane5to7, lane7to5])

coeff_5and7 = np.polyfit(lane5and7['Local_Y'], lane5and7['Local_X'], 1)
trend_5and7 = np.poly1d(coeff_5and7)
y_values_5and7 = np.linspace(-10, (mean_lane5and6 - 21.81) / (-0.02246), 100) # minimum x-value = -10

# Find the separation line between lanes 5 and 8 with linear trend line
lane5and8 = pd.concat([lane5to8, lane8to5])

coeff_5and8 = np.polyfit(lane5and8['Local_Y'], lane5and8['Local_X'], 1)
trend_5and8 = np.poly1d(coeff_5and8)
y_values_5and8 = np.linspace((mean_lane5and6 - 8.517) / 0.02219, 700, 100) # maximum x-value = 700

# Make file to save images with vehType
output_path_vehType = 'NGSIM_US-101_Trajectory_Data_frame_(vehType)/'
os.makedirs(output_path_vehType, exist_ok=True)

# Function to decide the vehicle type (color)
def vehTypeCheck(vehType):
    if vehType == 1: # Motorcycle
        return 'yellow'
    elif vehType == 2: # Auto
        return 'red'
    else: # Truck
        return 'green'
    
# Function to draw frame-by-frame images considering vehicle types
def drawImages_vehType(frame):
    plt.clf()  # initialize image

    data = df[df['Frame_ID']==frame]

    fig, ax =plt.subplots(figsize=(64, 4))

    # draw separated lines
    plt.axhline(y=mean_lane1and2, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane2and3, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane3and4, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane4and5, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane5and6, c='black', linestyle='--', zorder=1)

    plt.plot(y_values_5and7, trend_5and7(y_values_5and7), color='black', linestyle='--', zorder=1)
    plt.plot(y_values_5and8, trend_5and8(y_values_5and8), color='black', linestyle='--', zorder=1)

    # draw legend using color
    legend_patches = [
        patches.Patch(color='yellow', label='Motorcylce'),
        patches.Patch(color='red', label='Auto'),
        patches.Patch(color='green', label='Truck')
    ]

    # draw vehicles
    for index, row in data.iterrows():
        vehID = row['Vehicle_ID']
        width = row['Vehicle_Width']
        length = row['Vehicle_Length']
        pos_x = row['Local_Y']
        pos_y = row['Local_X']
        vehType = row['Vehicle_Class']
        color = vehTypeCheck(vehType)
        
        plt.text(pos_x-length/2, pos_y-width/2-0.5, f'{int(vehID)}', horizontalalignment='center', verticalalignment='center')
        
        # draw vehicle when lane changing
        if row['Movement'] > 0:
            lc_color = 'blue'
            lc_linewidth = 2
            plt.text(pos_x+2, pos_y, 'LC', fontweight='bold', verticalalignment='center', color='blue')   
        else:
            lc_color = 'black'
            lc_linewidth = 1
        
        ax.add_patch(
            patches.Rectangle(
                (pos_x-length, pos_y-width/2), 
                length, 
                width,
                linewidth = lc_linewidth,
                edgecolor = lc_color,
                facecolor = color, 
                fill=True,
                zorder=2)
        )

    ax.legend(handles=legend_patches, loc='upper right', ncol=3, fontsize=16, bbox_to_anchor=(1, 1.2))
        
    plt.ylim([-1.2,24.5])
    plt.xlim([-10,700])
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'NGSIM US-101 Trajectory Data (Vehicle Type) - frame {frame}', fontsize=20, y=1.05)

    plt.tight_layout()
    plt.savefig(f'{output_path_vehType}NGSIM_US101_vehType_{frame:05d}.png', bbox_inches='tight')

# Make file to save images with speed
output_path_speed = 'NGSIM_US-101_Trajectory_Data_frame_(Speed)/'
os.makedirs(output_path_speed, exist_ok=True)

# Function to draw frame-by-frame images considering speed (heatmap)
def drawImages_speed(frame):
    plt.clf()  # initialize image

    data = df[df['Frame_ID']==frame]

    fig, ax =plt.subplots(figsize=(64, 4))

    # draw separated lines
    plt.axhline(y=mean_lane1and2, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane2and3, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane3and4, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane4and5, c='black', linestyle='--', zorder=1)
    plt.axhline(y=mean_lane5and6, c='black', linestyle='--', zorder=1)

    plt.plot(y_values_5and7, trend_5and7(y_values_5and7), color='black', linestyle='--', zorder=1)
    plt.plot(y_values_5and8, trend_5and8(y_values_5and8), color='black', linestyle='--', zorder=1)

    # speed heatmap
    norm = mcolors.Normalize(vmin=0, vmax=40)
    cmap = cm.get_cmap('rainbow_r')
    
    # draw vehicles
    for index, row in data.iterrows():
        vehID = row['Vehicle_ID']
        width = row['Vehicle_Width']
        length = row['Vehicle_Length']
        pos_x = row['Local_Y']
        pos_y = row['Local_X']
        speed = row['Vehicle_Velocity']
        color = cmap(norm(speed))

        plt.text(pos_x-length/2, pos_y-width/2-0.5, f'{int(vehID)}', horizontalalignment='center', verticalalignment='center')

        if row['Movement'] > 0:
            lc_color = 'blue'
            lc_linewidth = 2
            plt.text(pos_x+2, pos_y, 'LC', fontweight='bold', verticalalignment='center', color='blue')   
        else:
            lc_color = 'black'
            lc_linewidth = 1

        ax.add_patch(
            patches.Rectangle(
                (pos_x-length, pos_y-width/2), 
                length, 
                width,
                linewidth = lc_linewidth,
                edgecolor = lc_color,
                facecolor = color, 
                fill=True,
                zorder=2
            )
        )

    plt.ylim([-1.2,24.5])
    plt.gca().invert_yaxis()
    plt.xlim([-10,700])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'NGSIM US-101 Trajectory Data (Speed) - frame {frame}', fontsize=20, y=1.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_path_speed}NGSIM_US101_Speed_{frame:05d}.png', bbox_inches='tight')

# draw and save images for each frame
for frame in range(df['Frame_ID'].max()):
    drawImages_vehType(frame)
    drawImages_speed(frame)

# Conversion image sequence to a video
# folder where image files saved
image_folder_vehType = 'NGSIM_US-101_Trajectory_Data_frame_(vehType)/'

# set output video path
output_video_path_vehType = 'NGSIM_US101_vehType.mp4'

# Sort iamge files and save them in a list
images = [img for img in sorted(os.listdir(image_folder_vehType)) if img.endswith(".png")]
# Get image size based on first image
first_img_path = os.path.join(image_folder_vehType, images[0])
with Image.open(first_img_path) as img:
    width, height = img.size

# convert image sequences to video
with imageio.get_writer(output_video_path_vehType, fps=10) as writer:
    for img_name in images:
        img_path = os.path.join(image_folder_vehType, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.resize((width, height))
                img_array = imageio.core.asarray(img)
            writer.append_data(img_array)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# folder where image files saved
image_folder_speed = 'NGSIM_US-101_Trajectory_Data_frame_(Speed)/'

# set output video path
output_video_path_speed = 'NGSIM_US101_Speed.mp4'

# Sort iamge files and save them in a list
images = [img for img in sorted(os.listdir(image_folder_speed)) if img.endswith(".png")]
# Get image size based on first image
first_img_path = os.path.join(image_folder_speed, images[0])
with Image.open(first_img_path) as img:
    width, height = img.size

# convert image sequences to video
with imageio.get_writer(output_video_path_speed, fps=10) as writer:
    for img_name in images:
        img_path = os.path.join(image_folder_speed, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.resize((width, height))
                img_array = imageio.core.asarray(img)
            writer.append_data(img_array)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")