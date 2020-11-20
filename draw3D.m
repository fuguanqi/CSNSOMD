[d1,d2,fc,k1k2_is_merged,k2k1_is_merged]=textread('outfile_8m.txt','%f %f %f %*f %*f %f %f','delimiter', ',');
m=[d1,d2,fc,k1k2_is_merged,k2k1_is_merged];
 index = find(m(:,4)==1|m(:,5)==1|m(:,3)==0);
% index = find(m(:,5)==0);
% display(index);
m(index,:)=[];
 scatter3(m(:,1),m(:,2),m(:,3),'.','MarkerEdgeColor','None','MarkerFaceColor','None'); 
% scatter3(m(:,1),m(:,2),m(:,3),'.','MarkerEdgeColor','yellow','MarkerFaceColor','yellow'); 
grid on
k= boundary(m(:,1),m(:,2),m(:,3),1);
xlim([0 100])
ylim([0 100])
zlim([0 1000])
hold on
trisurf(k,m(:,1),m(:,2),m(:,3),'FaceColor','none','LineStyle','-','FaceLighting','gouraud','EdgeColor','interp')

