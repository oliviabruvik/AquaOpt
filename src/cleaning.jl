using CSV, DataFrames

# Data downloaded from https://lusedata.hubocean.earth/
# Data limited to production area 5: Stadt til Hustadvika for 2012-2025
function load_and_clean(path::String)

    # Load data
    df = CSV.read(path, DataFrame)
    df = dropmissing(df)

    # Keep relevant columns
    keep_cols = [
        "uke", 
        "voksne_hunnlus", 
        "mekanisk_fjerning", 
        "year", 
        "lokalitetsnummer"
    ]
    df = df[:, keep_cols]

    # Translate column names to English
    new_cols = [
        "week",
        "adult_sealice",
        "mechanical_removal",
        "year",
        "location_number"
    ]
    rename!(df, new_cols)

    # Convert column types
    df.year = Int64.(df.year) # Previously Float64
    df.location_number = Int64.(df.location_number) # Previously Float64

    # Convert week and year to total weeks from start of monitoring
    df = convert_to_total_weeks(df)

    # Discretize sea lice levels
    df = discretize_sea_lice_levels(df)

    # Save to processed data folder
    CSV.write(joinpath("data", "processed", "sealice_data.csv"), df)

    return df
end

function convert_to_total_weeks(df::DataFrame)

    # Get first year of monitoring
    first_year = minimum(df.year)

    # Convert week and year to total weeks from start of monitoring
    df.total_week = (df.year .- first_year) .* 52 .+ df.week

    return df
end

function discretize_sea_lice_levels(df)
	df.adult_sealice = round.(df.adult_sealice, digits=1)
	return df
end