pip install gdown
LINK_TO_DATA=1s1svxg4CvyUi9eeMzqUbcpiwCtNc170z
mkdir test_data
cd test_data
gdown $LINK_TO_DATA -O test_data.zip
unzip test_data.zip
rm test_data.zip
