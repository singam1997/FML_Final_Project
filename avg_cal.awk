/---/{
	#print $0
	print prev t_sum/50
	prev = $0
	t_sum=0
}
$0 !~ /---/{
	#print $0 "NOT MATCHED"
	t_sum = t_sum+$0
}
end{
	print prev t_sum/50
}