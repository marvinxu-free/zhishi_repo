import pytablewriter
from tabulate import tabulate

def print_macdown_table(dataframe):
    writer = pytablewriter.MarkdownTableWriter()
    writer.from_dataframe(dataframe=dataframe)
    writer.write_table()

def print_table(dateframe):
    print(tabulate(tabular_data=dateframe,tablefmt='fancy_grid'))

