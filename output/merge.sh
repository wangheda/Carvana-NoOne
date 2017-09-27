
ensemble="../tools/ensemble"
cat header.tpl > all.mergecsv

for file in *.csv; do
  if [ ! -f ${file}.body ]; then
    tail -n 100064 $file > ${file}.body
  fi
done

$ensemble *.csv.body >> all.mergecsv

