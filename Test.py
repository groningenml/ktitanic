# Set-up virtual environment
# py -3 -m venv env
# env\scripts\activate

1
while read requirements.txt; do conda install --yes $requirement; done < requirements.txt

# Jupyter notebook
#%%
msg = "Hello GroningenML"
print(msg)

#%% [markdown]