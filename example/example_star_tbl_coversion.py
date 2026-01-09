from utils.io_dynamo import dynamo_tbl_vll_to_relion_star,relion_star_to_dynamo_tbl,read_vll_to_df,save_sorted_vll_by_tomonames
import os

project_dir = '/Users/muwang/Desktop/test'

star_ori = os.path.join(project_dir,'input','refined_table_ref_001_ite_0004_tomo_name.star')
star_new = os.path.join(project_dir,'output','refined_table_ref_001_ite_0004_tomo_name_new.star')

tbl_output = os.path.join(project_dir,'output','refined_table_ref_001_ite_0004_tomo_name.tbl')
vll_output = os.path.join(project_dir,'output','refined_table_ref_001_ite_0004_tomo_name.vll')

# convert star file to tbl file
df =relion_star_to_dynamo_tbl(
    star_ori, 
    6.72,
    tomogram_size=(999,999,499), 
    output_file=tbl_output)

# vll file re-orgnization
micrograph_names = df['rlnMicrographName'].unique()
# save micrograph names to vll file
# write a new vll file with micrograph names
with open(vll_output, 'w') as f:
    for name in micrograph_names:
        f.write(name + '\n')
#vll_df = read_vll_to_df(vll_output)
#save_sorted_vll_by_tomonames(micrograph_names, vll_df, vll_output)

# convert tbl, vll file to star file
star_new = star_new.replace('.star', '_new.star')
df = dynamo_tbl_vll_to_relion_star(tbl_output, vll_output, output_file=star_new,pixel_size=6.72,tomogram_size=(999,999,499),output_centered=True)



