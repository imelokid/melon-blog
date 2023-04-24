WORKSPACE=/Users/melonkid/workspace/myself
PROGRAM_NAME="melon-blog"

if [ ! -n "$1" ] ;then
    echo 'file name is blank'
else
    scp -r $WORKSPACE/$PROGRAM_NAME/public/$1 melonkid@49.232.131.132:~/$PROGRAM_NAME
fi


