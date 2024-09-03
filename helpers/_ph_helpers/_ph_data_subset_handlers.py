# %%
import os
import pandas as pd
import glob
import shutil

CONST_META_COLS = [
    "composer",
    "title",
    "split",
    "version",
    "folder",
    "xml_score",
    "midi_score",
    "midi_performance",
    "audio_performance",
    "midi_path",
    "audio_path",
    "duration_sec",
    "robust_note_alignment",
]


def get_leaf_directories(root_dir):
    leaf_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            leaf_directories.append(dirpath)
    return leaf_directories



def restructure_files(directory, subset='asap_maestro'):
  
    if subset == 'asap_maestro':
        leaf_dirs = get_leaf_directories(directory)
        # TMP
        leaf_dirs = [d for d in leaf_dirs if not d.endswith('disklavier') and not d.endswith('maestro')]
        
        for leaf_dir in leaf_dirs:
            
            files = os.listdir(leaf_dir)
            files = [f for f in files if not f.startswith('.')]
            
            performers = [f.split('.')[0] for f in files if f.endswith('.mid') and not f.startswith('kong') and not f.startswith('oaf') and not f.startswith('T5')]
            
            for performer in performers:
                
                for file in sorted(files):
                    if performer in file:
                        
                        if file in [f'{performer}.mid', f'{performer}.match', 'midi_score.midi'] or file.endswith('musicxml') or file.endswith('csv'):
                            continue
                        else:
                            if file.split('.')[0].endswith('disklavier') or file.split('.')[0].endswith('disklavier_16bit'):
                                new_path = os.path.join(leaf_dir, f'{performer}_disklavier')
                            else:
                                new_path = os.path.join(leaf_dir, f'{performer}_maestro')
                            
                            if not os.path.exists(new_path):
                                    os.makedirs(new_path)
                        
                            shutil.move(os.path.join(leaf_dir, file), os.path.join(new_path, file))
                    
    
    else:
        piece_dirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
        for piece_dir in piece_dirs:
            files = os.listdir(os.path.join(directory, piece_dir))
            files = [f for f in files if not f.startswith('.')]
            files = [f for f in files if not os.path.isdir(os.path.join(directory, piece_dir, f))]
            
            for file in files:
                if file == f'{piece_dir}.musicxml' or file == f'{piece_dir}.mid' or file == f'{piece_dir}.match' or file.endswith('csv'):
                    continue

                else:
                    if 'disklavier' in file:
                        new_path = os.path.join(directory, piece_dir, 'batik_disklavier')
                    else:
                        new_path = os.path.join(directory, piece_dir, 'batik_audio')
                
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    
                    shutil.move(os.path.join(directory, piece_dir, file), os.path.join(new_path, file))
    
    return None
        
asap_maestro_subset = '/Users/huispaty/Code/python/tri24_local/data/asap_maestro_eval_subset'
# restructure_files(asap_maestro_subset)

# batik_mozart_subset = '/Users/huispaty/Code/python/tri24_local/data/batik_mozart_subset'
# restructure_files(batik_mozart_subset, subset='batik_mozart')






#%%
def handle_files(
    directory,
    type="rename",
    prefix=None,
    suffix=None,
    file_ending=None,
    new_root = None,
    replace=("old", "new"),
):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix) and file.endswith(file_ending):
                if type == "rename":
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, file.replace(replace[0], replace[1]))
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                elif type == "remove":
                    os.remove(os.path.join(root, file))
                    print(f"Deleted {file}")
                elif type == "move":
                    old_path = os.path.join(root, file)
                    # TODO : change according to path structure
                    new_subdir_path = '/'.join(old_path.split(directory)
                                               [1].split('/')[2:])
                    new_path = os.path.join(new_root, new_subdir_path)
                    # os.replace(old_path, new_path)
                    

def create_meta_csv(pieces, subset='batik_mozart'):

    meta_csv = pd.DataFrame(columns=CONST_META_COLS)

    if subset == 'batik_mozart':
        for i, piece in enumerate(pieces):
            meta_csv.at[i, 'composer'] = 'Mozart'
            meta_csv.at[i, 'title'] = piece
            meta_csv.at[i, 'split'] = None
            meta_csv.at[i, 'version'] = None
            meta_csv.at[i, 'folder'] = piece
            meta_csv.at[i, 'xml_score'] = f'{piece}.musicxml'
            meta_csv.at[i, 'midi_score'] = None
            meta_csv.at[i, 'midi_performance'] = f'{piece}.mid'
            meta_csv.at[i, 'audio_performance'] = f'{piece}.wav'
            meta_csv.at[i, 'midi_path'] = None
            meta_csv.at[i, 'audio_path'] = None
            meta_csv.at[i, 'duration_sec'] = None # TODO
            meta_csv.at[i, 'robust_note_alignment'] = 1

    return meta_csv

new_path = "/Users/huispaty/Code/python/tri24_local/data/"
path = "/Users/huispaty/Code/python/tri24_local/"

# handle_files(path, type='move', prefix='T5',
            #  file_ending='.mid', new_root=new_path)


# handle_files(path, type='remove', prefix='_tmp', file_ending='.csv')
# handle_files(path, prefix='measure_', file_ending='.csv', replace=('measure_', '_tmp'))
# handle_files(path, prefix='piecewise_', file_ending='.csv',
#              replace=('piecewise_', '_tmp_piecewise'))

# batik_mozart_meta = create_meta_csv(['kv332_1', 'kv332_3'])
# batik_mozart_meta.to_csv('/Users/huispaty/Code/python/tri24_local/data/meta_csv/meta_batik_mozart_subset.csv', index=False)
