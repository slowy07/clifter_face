def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CUstomDatasetDataLoader

    data_loader = CustomDatasetDataLoaderdef()
    print(data_loader.name())
    return data_loader
