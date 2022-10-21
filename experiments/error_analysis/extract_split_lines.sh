#get all section names/titles at the beginning of our splits. Execute from within a directory containing the preprocessed (split) data files
grep -E -h '^<([^>]+)>[^<]+</\1>' *.txt
# to also get the file names:
#grep -E '^<([^>]+)>[^<]+</\1>' *.txt