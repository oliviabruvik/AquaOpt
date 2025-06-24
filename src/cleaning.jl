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
    # df = discretize_sea_lice_levels(df)

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

function load_baretswatch_lice_data(lice_path::String)

    # Load data
    lice_df = CSV.read(lice_path, DataFrame)

    # Keep relevant columns for lice data
    lice_keep_cols = [
        "Uke", 
        "År",
        "Lokalitetsnummer",
        "Voksne hunnlus",
        "Har telt lakselus",
        "Over lusegrense uke",
        "Sjøtemperatur"
    ]
    lice_df = lice_df[:, lice_keep_cols]

    # Translate column names to English
    lice_new_cols = [
        "week",
        "year",
        "site_number",
        "adult_sealice",
        "has_counted_lice",
        "over_lice_limit_week",
        "sea_temperature"
    ]
    rename!(lice_df, lice_new_cols)

    # Drop rows where Har telt is not Ja
    lice_df = lice_df[lice_df.has_counted_lice .== "Ja", :]

    return lice_df
end

function load_baretswatch_disease_data(disease_path::String)

    # Load data
    disease_df = CSV.read(disease_path, DataFrame)

    # Keep relevant columns for disease data
    disease_keep_cols = [
        "Uke",
        "År",
        "Lokalitetsnummer",
        "Sykdom",
        "Status",
        "Fra dato",
        "Til dato"
    ]
    disease_df = disease_df[:, disease_keep_cols]

    # Translate column names to English
    disease_new_cols = [
        "week",
        "year",
        "site_number",
        "disease",
        "status",
        "start_date",
        "end_date"
    ]
    rename!(disease_df, disease_new_cols)

    return disease_df
end

function load_baretswatch_treatment_data(treatment_path::String)

    # Load data
    treatment_df = CSV.read(treatment_path, DataFrame)

    # Keep relevant columns for treatment data
    treatment_keep_cols = [
        "Uke",
        "År",
        "Lokalitetsnummer",
        "Tiltak",
        "Type behandling",
    ]
    treatment_df = treatment_df[:, treatment_keep_cols]

    # Translate column names to English
    treatment_new_cols = [
        "week",
        "year",
        "site_number",
        "treatment",
        "treatment_type"
    ]
    rename!(treatment_df, treatment_new_cols)

    return treatment_df
end

function combine_baretswatch_data(lice_df, disease_df, treatment_df)

    # Combine inner data by week, year, and site number
    inner_df = innerjoin(lice_df, disease_df, on=[:week, :year, :site_number])
    inner_df = innerjoin(inner_df, treatment_df, on=[:week, :year, :site_number])

    # Combine outer data by week, year, and site number
    outer_df = outerjoin(lice_df, disease_df, on=[:week, :year, :site_number])
    outer_df = outerjoin(outer_df, treatment_df, on=[:week, :year, :site_number])

    return inner_df, outer_df
end

function translate_df_to_english(df::DataFrame)

    # Translate Ja to true and Nei to false
    df.has_counted_lice = coalesce.(df.has_counted_lice .== "Ja", false)
    df.over_lice_limit_week = coalesce.(df.over_lice_limit_week .== "Ja", false)

    # Handle missing values and convert disease column to binary
    df.treatment_type = coalesce.(df.treatment_type, "None")
    df.treatment = coalesce.(df.treatment, "None")
    df.disease = coalesce.(df.disease, "None")

    # Translate treatment types to english
    df.treatment = ifelse.(df.treatment .== "ikke-medikamentell", "non-medicinal", df.treatment)
    df.treatment = ifelse.(df.treatment .== "medikamentell", "medicinal", df.treatment)
    df.treatment_type = ifelse.(df.treatment_type .== "badebehandling", "bath_treatment", df.treatment_type)
    df.treatment_type = ifelse.(df.treatment_type .== "termisk behandling", "thermal_treatment", df.treatment_type)
    df.treatment_type = ifelse.(df.treatment_type .== "mekanisk behandling", "mechanical_treatment", df.treatment_type)
    df.treatment_type = ifelse.(df.treatment_type .== "ferskvannsbehandling", "freshwater_treatment", df.treatment_type)
    df.treatment_type = ifelse.(df.treatment_type .== "fôrbehandling", "feed_treatment", df.treatment_type)
    df.treatment_type = ifelse.(df.treatment_type .== "annen behandling", "other_treatment", df.treatment_type)

    return df
end

function clean_bayesian_data(df::DataFrame)

    # Sort by site number, year, then week
    df = sort(df, [:site_number, :year, :week])
    
    # Translate to english
    df = translate_df_to_english(df)

    # Convert column types
    df.year = Int64.(df.year) # Previously Float64
    df.site_number = Int64.(df.site_number) # Previously Float64

    # Convert week and year to total weeks from start of monitoring
    df = convert_to_total_weeks(df)

    # Save to processed data folder
    CSV.write(joinpath("data", "processed", "bayesian_data.csv"), df)

    return df
end

function process_bayesian_data()

    LICE_PATH = "data/raw/salmon_lice.csv"
    DISEASE_PATH = "data/raw/fish_disease.csv"
    TREATMENT_PATH = "data/raw/lice_treatments.csv"

    # Load data
    lice_df = load_baretswatch_lice_data(LICE_PATH)
    disease_df = load_baretswatch_disease_data(DISEASE_PATH)
    treatment_df = load_baretswatch_treatment_data(TREATMENT_PATH)

    # Combine data
    inner_df, outer_df = combine_baretswatch_data(lice_df, disease_df, treatment_df)

    # Prepare data for Bayesian analysis
    inner_df = clean_bayesian_data(inner_df)
    outer_df = clean_bayesian_data(outer_df)

    # Save to processed data folder
    CSV.write(joinpath("data", "processed", "bayesian_inner_data.csv"), inner_df)
    CSV.write(joinpath("data", "processed", "bayesian_outer_data.csv"), outer_df)

end

function main()
    load_and_clean("data/raw/licedata.csv")
end

main()