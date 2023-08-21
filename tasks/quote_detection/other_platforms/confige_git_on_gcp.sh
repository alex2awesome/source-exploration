# on local
gcloud compute config-ssh
gcp_ssh=instance-6.us-central1-a.usc-research

# on gcp
ssh $gcp_ssh
git init --bare bare_project.git
echo "git --work-tree=deployed_project --git-dir=bare_project.git checkout -f" >> bare_project.git/hooks/post-receive
sudo chmod +x bare_project.git/hooks/post-receive

# on local
git remote add gcp_test "$gcp_ssh:~/bare_project.git"
git push --set-upstream gcp_test master

# to change
git remote set-url gcp_test "$gcp_ssh:~/bare_project.git"