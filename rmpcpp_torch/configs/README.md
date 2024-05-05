# Parameters

There are 5 parameter file types in the training and mini-config folder:
- General parameters
- Model parameters
- Data pipeline train parameters
- Data pipeline validation parameters
- Probabilistic worlds parameters

For the model and data pipeline parameters there can be multiple parameter files inside that folder. Only 1 will be selected during training by passing the name of the parameter file to the parser. See the parser in the rmp_dl python package for how that is done. Furthermore, selection of the right overall folder (e.g. training vs mini-config) is also done by passing the correct folder to the parser. 

These files can reference parameters from each other. E.g. the `planner/dt` parameter inside `general_parameters.yml` can be accessed from any of the other (non-autoencoder) files (and also from other blocks within `generator_parameters.yml`) by `$general/planner/dt`. I.e. you need to prepend the file with `$FILECODE`, and then use foward slashes to jumpy into blocks. The naming of the files are as such:

- `$general/`
- `$model/`
- `$datatrain/`
- `$dataval/`
- `$worlds/`

Note that you SHOULD NOT reference variables in other files from the general params file, as this is not supported in all parts of the code (some parts of the code expect `$general` to be self contained. )

See `rmp_io.ConfigUtil._resolve_configs` on how references are resolved exactly. 
