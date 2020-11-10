[d1,d2,fc,k1k2_is_merged,k2k1_is_merged]=textread('outfile.txt','%f %f %f %*f %*f %f %f','delimiter', ',');
m=[d1,d2,fc,k1k2_is_merged,k2k1_is_merged];
index = find(m(:,4)==1);
% index = find(m(:,5)==0);
% display(index);
m(index,:)=[];
scatter3(m(:,1),m(:,2),m(:,3),'.'); 
grid on
k= boundary(m(:,1),m(:,2),m(:,3),1);
hold on
trisurf(k,m(:,1),m(:,2),m(:,3),'FaceColor','blue','FaceAlpha',0.1,'LineStyle','none')

