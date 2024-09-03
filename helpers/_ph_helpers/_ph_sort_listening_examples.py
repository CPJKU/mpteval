#%%
import os
import shutil
import pandas as pd





def sort_disklavier_eval_data(disklavier_data_path, eval_data_root):
    '''
    Move disklavier recordings to the eval data root
    '''
    for root, dirs, files in os.walk(disklavier_data_path):
        for file in files:
            # file name has suffic '_v1' from piano capture
            if file.endswith('_v1.wav'):
                # construct the source and destination paths
                src_path = os.path.join(root, file)
                dest_path = os.path.join(eval_data_root, os.path.relpath(
                    root, disklavier_data_path), file.replace('_v1', '_disklavier'))

                # create the destination path if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)
                print(f"Moved '{src_path}' to '{dest_path}'")




def sort_listening_excerpts(listening_excerpts_csv, src_dir, dest_dir):
    excerpts_df = pd.read_csv(listening_excerpts_csv)
    print('df columns:', excerpts_df.columns.values)
    print('num unique pieces:', excerpts_df['canonical_title'].nunique())
    print('num unique composers:', excerpts_df['canonical_composer'].nunique())
    print(f'Sorting to musical dims: {excerpts_df["attribute_compact"].unique()}')
    
    excerpts_df = excerpts_df.sort_values(by='excerpt_filenamen')
    
    for index, row in excerpts_df.iterrows():
        mus_dim = str(row['attribute_compact'])
        excerpt_fn = str(row['excerpt_filename'])

        # create mus_dim dir
        mus_dim_path = os.path.join(dest_dir, mus_dim)
        if not os.path.exists(mus_dim_path):
            os.makedirs(mus_dim_path)

        # get source and destination
        src_file = os.path.join(src_dir, excerpt_fn)
        if not os.path.exists(src_file):
            print(f'{index} - missing {excerpt_fn}')
        dst_file = os.path.join(mus_dim_path, excerpt_fn)
        # move
        shutil.move(src_file, dst_file)
        print(f"Moved {excerpt_fn} to {mus_dim_path}")
    
    return None


# listening_excerpts_csv = '/share/hel/home/patricia/Research/transcription/tri24/helpers/listening_excerpts_sorted_cleaned.csv'
# src_path = '/share/hel/home/patricia/Research/transcription/tri24/data/tmp--LISTENING-EXAMPLES' 
# dst_path = '/share/hel/home/patricia/Research/transcription/tri24/data/audio'

# sort_listening_excerpts(listening_excerpts_csv, src_path, dst_path)
