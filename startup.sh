# find hexo process
hexo_pid=`ps -ef|grep hexo|grep -v grep|awk '{print $2}'`

#kill old process
echo "kill hexo_pid["$hexo_pid"]"
for id in $hexo_pid
  do
    kill -9 $id
  done

# start up hexo
hexo s -g & >log.txt

sleep 5

hexo_pid=`ps -ef|grep hexo|grep -v grep|awk '{print $2}'`
echo $hexo_pid
